import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import HEADS

# ==============================================================================
# Helper Modules: Positional Encodings & Time Embeddings
# ==============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard PE to provide sequence order information to the Transformer."""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Batch, SeqLen, Dim]
        return x + self.pe[:, :x.size(1)]

class TimeEmbedding(nn.Module):
    """Embeds scalar t into a vector."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)

# ==============================================================================
# DiT Block (Diffusion Transformer Layer)
# ==============================================================================

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block using AdaLN-Zero.
    1. Self-Attention (Trajectory Consistency)
    2. Cross-Attention (Conditioning on LLM Context)
    3. AdaLN (Time Conditioning)
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.0):
        super().__init__()
        # Norms are affine=False because scale/shift is predicted by AdaLN
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # AdaLN Modulation: Regresses 6 parameters per channel from time embedding
        # (scale_sa, shift_sa, scale_ca, shift_ca, scale_ffn, shift_ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        
        # Zero-init: Starts training as an identity block (Crucial for DiT stability)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, t_emb, key_padding_mask=None, attn_mask=None):
        """
        x: Noisy Trajectory [Batch, N_points, hidden_size]
        c: LLM Context      [Batch, N_tokens, hidden_size]
        t_emb: Time Embed   [Batch, hidden_size]
        key_padding_mask:   [Batch, N_tokens] (True = Pad/Ignore, False = Valid)
        """
        # 1. Regress Modulation Parameters
        shift_scale = self.adaLN_modulation(t_emb)
        scale_sa, shift_sa, scale_ca, shift_ca, scale_ffn, shift_ffn = shift_scale.chunk(6, dim=1)

        # 2. Self-Attention (Trajectory Consistency)
        x_skip = x
        x = self.norm1(x)
        x = x * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x_skip + x

        # 3. Cross-Attention (Conditioning on LLM)
        x_skip = x
        x = self.norm2(x)
        x = x * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1)
        
        # Pass the mask here to ignore padding tokens from LLM
        x, _ = self.cross_attn(
            query=x, 
            key=c, 
            value=c, 
            key_padding_mask=key_padding_mask
        )
        x = x_skip + x

        # 4. Feed Forward
        x_skip = x
        x = self.norm3(x)
        x = x * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1)
        x = self.ffn(x)
        x = x_skip + x

        return x

# ==============================================================================
# Main Head Class
# ==============================================================================

@HEADS.register_module()
class DiffusionTransformerTrajectoryHead(BaseModule):
    """
    Transformer-based Flow Matching Head.
    Supports FP16 inference and FP32 Loss calculation.
    """
    def __init__(self,
                 hidden_size: int = 2048,    # Input Dim (LLM Output)
                 transformer_dim: int = 1024, # Internal Dim of DiT
                 num_path_points: int = 6,
                 num_output_points: int = None,
                 path_coord_dim: int = 2,
                 num_inference_steps: int = 10,
                 flow_sig_min: float = 1e-4,
                 flow_sampling: str = "beta",
                 flow_alpha: float = 1.5,
                 flow_beta: float = 1.0,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 normalize_mean: list = [[2.51, -0.01], [5.02, -0.03], [7.53, -0.06], [10.02, -0.08], [12.50, -0.11], [14.96, -0.13]],
                 normalize_std: list = [[1.81, 0.05], [3.60, 0.20], [5.40, 0.45], [7.18, 0.80], [8.95, 1.24], [10.71, 1.75]],
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)

        self.fp16_enabled = True # Use auto_fp16 decorator
        self.num_path_points = num_path_points
        # num_output_points controls output trajectory length; defaults to num_path_points.
        # Set > num_path_points to predict a longer horizon without re-caching LLM features.
        self.num_output_points = num_output_points if num_output_points is not None else num_path_points
        self.path_coord_dim = path_coord_dim
        self.transformer_dim = transformer_dim
        
        # Normalization buffers
        self.register_buffer('traj_mean', torch.tensor(normalize_mean).float())
        self.register_buffer('traj_std', torch.tensor(normalize_std).float())

        # Sampling Configuration
        self.random_init = False  # set True at inference time to start from N(0,1) instead of zeros
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.flow_sampling = flow_sampling
        if self.flow_sampling == "beta":
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        # 1. Projections: 2-layer MLP from LLM hidden size to the DiT width
        self.context_proj = nn.Sequential(
        nn.Linear(hidden_size, hidden_size//2),      
        nn.LayerNorm(hidden_size//2),                
        nn.SiLU(),                         
        nn.Linear(hidden_size//2, transformer_dim)   
        )
        # Map [x, y] to Transformer Dim
        self.input_proj = nn.Linear(path_coord_dim, transformer_dim)
        self.pos_embed = SinusoidalPositionalEncoding(transformer_dim, self.num_output_points)
        self.time_embedding = TimeEmbedding(transformer_dim)

        # 2. Transformer Backbone (DiT)
        self.blocks = nn.ModuleList([
            DiTBlock(transformer_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # 3. Output
        self.final_norm = nn.LayerNorm(transformer_dim)
        self.final_proj = nn.Linear(transformer_dim, path_coord_dim) 

    def _sample_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample timestep t with correct dtype."""
        if self.flow_sampling == "uniform":
            grid = torch.arange(batch_size, device=device, dtype=dtype) / batch_size
            offset = torch.rand(1, device=device, dtype=dtype)
            t = (grid + offset) % (1 - 1e-5)
        elif self.flow_sampling == "beta":
            t = self.flow_beta_dist.sample((batch_size,)).to(device=device, dtype=dtype)
        else:
            t = torch.rand(batch_size, device=device, dtype=dtype)
        return t
    
    def _generate_causal_mask(self, sz, device):
        """
        Generates a mask where position i can only attend to positions 0...i
        Returns: [sz, sz] float mask
        """
        # triu = Upper Triangular. We want to mask the upper triangle (future).
        # We start from diagonal=1 (so token i sees itself).
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        
        # Replace 1s (future) with -inf, 0s (past/present) with 0.0
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _predict_velocity(self, llm_features: torch.Tensor, noisy_path: torch.Tensor, t: torch.Tensor, context_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the DiT.
        llm_features: [B, Seq, Hidden]
        noisy_path:   [B, 6, 2]
        t:            [B]
        context_mask: [B, Seq] (1=Valid, 0=Pad)
        """
        # Safety cast: Ensure inputs match weight dtype (handle FP16/FP32 mismatches)
        target_dtype = self.input_proj.weight.dtype
        if noisy_path.dtype != target_dtype:
            noisy_path = noisy_path.to(target_dtype)
        if llm_features.dtype != target_dtype:
            llm_features = llm_features.to(target_dtype)
        if t.dtype != target_dtype:
            t = t.to(target_dtype)

        # 1. Embed Query (Trajectory)
        x = self.input_proj(noisy_path) # [B, 6, 256]
        x = self.pos_embed(x)           # Add sequence order info

        #create causal mask
        seq_len = x.shape[1]
        causal_mask = self._generate_causal_mask(seq_len, x.device)

        # 2. Embed Time
        t_emb = self.time_embedding(t.view(-1, 1)) # [B, 256]

        # 3. Embed Context (Project LLM features once)
        c = self.context_proj(llm_features)

        # 4. Handle Attention Mask
        key_padding_mask = None
        if context_mask is not None:
            # PyTorch MultiheadAttention expects: True = Pad (Ignore), False = Keep
            # Input context_mask is: 0 = Pad, 1 = Keep
            # So we invert the logic: (mask == 0) -> True
            if context_mask.shape[0] == x.shape[0]:
                key_padding_mask = (context_mask == 0).to(x.device)

        # 5. Transformer Layers
        for block in self.blocks:
            x = block(x, c=c, t_emb=t_emb, key_padding_mask=key_padding_mask, attn_mask=causal_mask)

        # 6. Output
        x = self.final_norm(x)
        velocity = self.final_proj(x) # [B, 6, 2]
        return velocity

    @auto_fp16(apply_to=('driving_output_features',))
    def forward(self, driving_output_features: torch.Tensor, context_mask: torch.Tensor = None, **kwargs):
        """
        Standard interface.
        driving_output_features: Last-layer hidden states of the LLM [B, Seq, Hidden].
            During training, answer tokens are already zeroed out by the detector.
        context_mask: The mask indicating valid LLM tokens (1 = valid, 0 = ignore);
            used as key padding mask in the cross-attention layers.
        """
        dtype = driving_output_features.dtype
        device = driving_output_features.device

        # Inference Mode
        if not self.training:
            predicted_path = self.inference_euler(driving_output_features, context_mask)
            return {'predicted_path': predicted_path}

        # Training Mode
        batch_size = driving_output_features.shape[0]

        # Sample t and noise in correct dtype
        t = self._sample_t(batch_size, device, dtype)
        x0_noise = torch.randn(batch_size, self.num_output_points, self.path_coord_dim, device=device, dtype=dtype)

        return {
            'llm_features': driving_output_features,
            'context_mask': context_mask,
            'x0_noise': x0_noise,
            't': t
        }

    def loss(self,
             gt_path: torch.Tensor,
             gt_path_valid: torch.Tensor,
             preds_dicts: dict,
             **kwargs) -> dict:
        """
        Calculate Flow Matching Loss.
        CRITICAL: Upcasts to FP32 for numerical stability.
        """
        llm_features = preds_dicts['llm_features']
        context_mask = preds_dicts.get('context_mask', None)
        x0_noise = preds_dicts['x0_noise']
        t = preds_dicts['t']
        
        # 1. Normalize GT (using Model dtype first)
        dtype_model = llm_features.dtype 
        mean_model = self.traj_mean.to(device=gt_path.device, dtype=dtype_model)
        std_model = self.traj_std.to(device=gt_path.device, dtype=dtype_model)
        gt_path_model = gt_path.to(dtype_model)
        gt_path_norm_model = (gt_path_model - mean_model) / (std_model + 1e-6)
        
        # 2. Flow Matching Interpolation
        t_reshaped = t.view(-1, 1, 1)
        psi_t_path = (1 - (1 - self.flow_sig_min) * t_reshaped) * x0_noise + t_reshaped * gt_path_norm_model
        
        # 3. Predict Vector Field (Run Network in FP16/BF16)
        predicted_velocity = self._predict_velocity(llm_features, psi_t_path, t, context_mask=context_mask)

        # 4. Loss Calculation (Upcast to FP32)
        predicted_velocity = predicted_velocity.float()
        t_float = t.float().view(-1, 1, 1)
        x0_noise_float = x0_noise.float()
        gt_path_float = gt_path.float()
        
        mean_float = self.traj_mean.to(device=gt_path.device, dtype=torch.float32)
        std_float = self.traj_std.to(device=gt_path.device, dtype=torch.float32)
        gt_path_norm_float = (gt_path_float - mean_float) / (std_float + 1e-6)

        # Target Velocity: u_t(x) = x1 - (1-sigma)*x0
        target_velocity = gt_path_norm_float - (1 - self.flow_sig_min) * x0_noise_float
        
        # MSE
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        
        # Mask invalid GT (padding in ground truth)
        per_waypoint_mask = (torch.abs(gt_path_float).sum(dim=-1) > 1e-4).float().unsqueeze(-1)
        masked_loss = loss * per_waypoint_mask
        final_loss = masked_loss.sum() / (per_waypoint_mask.sum() + 1e-6)

        return {'loss_traj_fm': torch.nan_to_num(final_loss)}

    def inference_euler(self, llm_features: torch.Tensor, context_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate trajectory using Euler integration.
        """
        batch_size = llm_features.shape[0]
        device = llm_features.device
        dtype = llm_features.dtype

        if self.random_init:
            path = torch.randn((batch_size, self.num_output_points, self.path_coord_dim), device=device, dtype=dtype)
        else:
            path = torch.zeros((batch_size, self.num_output_points, self.path_coord_dim), device=device, dtype=dtype)
        num_steps = self.num_inference_steps
        dt = 1.0 / num_steps
        
        # Integration Loop
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=dtype)

            # Predict velocity (vector field)
            velocity = self._predict_velocity(llm_features, path, t, context_mask=context_mask)

            # Step
            path = path + dt * velocity

        # Denormalize
        mean = self.traj_mean.to(device=device, dtype=dtype)
        std = self.traj_std.to(device=device, dtype=dtype)
        return (path * std) + mean

    #second-order inference
    def inference_rk4(self, llm_features: torch.Tensor, context_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate trajectory using Runge-Kutta 4 (RK4) integration.
        Much more precise than Euler for curved trajectories.
        """
        batch_size = llm_features.shape[0]
        device = llm_features.device
        dtype = llm_features.dtype

        # Start from Mean of Prior (Zeros) - Best for ADE
        path = torch.zeros((batch_size, self.num_output_points, self.path_coord_dim), device=device, dtype=dtype)

        dt = 1.0 / self.num_inference_steps
        
        for i in range(self.num_inference_steps):
            t_curr = i * dt
            
            # 1. k1 (Current slope)
            t = torch.full((batch_size,), t_curr, device=device, dtype=dtype)
            k1 = self._predict_velocity(llm_features, path, t, context_mask=context_mask)
            
            # 2. k2 (Midpoint slope using k1)
            t_mid = torch.full((batch_size,), t_curr + 0.5 * dt, device=device, dtype=dtype)
            k2 = self._predict_velocity(llm_features, path + 0.5 * dt * k1, t_mid, context_mask=context_mask)
            
            # 3. k3 (Midpoint slope using k2)
            k3 = self._predict_velocity(llm_features, path + 0.5 * dt * k2, t_mid, context_mask=context_mask)
            
            # 4. k4 (End slope using k3)
            t_end = torch.full((batch_size,), t_curr + dt, device=device, dtype=dtype)
            k4 = self._predict_velocity(llm_features, path + dt * k3, t_end, context_mask=context_mask)
            
            # Combine
            path = path + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Denormalize
        mean = self.traj_mean.to(device=device, dtype=dtype)
        std = self.traj_std.to(device=device, dtype=dtype)
        return (path * std) + mean

    def get_trajectories(self, preds_dicts: dict, **kwargs):
        """Extracts trajectories for evaluation/saving."""
        predicted_path = preds_dicts['predicted_path']
        return [path.detach().cpu().numpy() for path in predicted_path]
