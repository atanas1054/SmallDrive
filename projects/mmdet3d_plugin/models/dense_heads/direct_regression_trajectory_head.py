import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import HEADS


def _masked_mean_pool(features: torch.Tensor, context_mask: torch.Tensor = None) -> torch.Tensor:
    """Pool full LLM context while ignoring masked tokens."""
    features = torch.nan_to_num(features)

    if context_mask is None:
        return features.mean(dim=1)

    if context_mask.dim() > 2:
        context_mask = context_mask.squeeze(-1)

    mask = context_mask.to(device=features.device, dtype=features.dtype)
    mask = mask.unsqueeze(-1)
    masked_features = features * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked_features.sum(dim=1) / denom


def _gt_valid_mask(gt_path: torch.Tensor, gt_path_valid: torch.Tensor) -> torch.Tensor:
    """Normalize gt_path_valid into a [B, W, 1] float mask."""
    batch_size, num_points = gt_path.shape[:2]
    device = gt_path.device

    if gt_path_valid is None:
        mask = torch.ones(batch_size, num_points, device=device, dtype=torch.float32)
    else:
        mask = gt_path_valid.to(device=device)
        if mask.dim() == 0:
            mask = mask.view(1, 1).expand(batch_size, num_points)
        elif mask.dim() == 1:
            if mask.numel() == batch_size:
                mask = mask.view(batch_size, 1).expand(batch_size, num_points)
            elif mask.numel() == num_points:
                mask = mask.view(1, num_points).expand(batch_size, num_points)
        elif mask.dim() == 2:
            if mask.shape[1] == 1:
                mask = mask.expand(batch_size, num_points)
            elif mask.shape[0] == 1:
                mask = mask.expand(batch_size, num_points)
        else:
            mask = mask.squeeze(-1)
            if mask.dim() == 1:
                mask = mask.view(batch_size, 1).expand(batch_size, num_points)

    return mask.to(dtype=torch.float32).unsqueeze(-1)


def _path_to_displacements(path: torch.Tensor) -> torch.Tensor:
    shifted_path = torch.zeros_like(path)
    shifted_path[:, 1:, :] = path[:, :-1, :]
    return path - shifted_path


class _DirectRegressionTrajectoryHead(BaseModule):
    """Shared direct-supervision behavior for full-context trajectory ablations."""

    def __init__(
        self,
        hidden_size: int = 2048,
        num_path_points: int = 6,
        path_coord_dim: int = 2,
        loss_traj: dict = None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.fp16_enabled = True
        self.hidden_size = hidden_size
        self.num_path_points = num_path_points
        self.path_coord_dim = path_coord_dim

        loss_traj = loss_traj or {}
        self.loss_beta = loss_traj.get('beta', 1.0)
        self.loss_weight = loss_traj.get('loss_weight', 1.0)

    def _match_parameter_dtype(self, features: torch.Tensor) -> torch.Tensor:
        target_dtype = next(self.parameters()).dtype
        if features.dtype != target_dtype:
            features = features.to(target_dtype)
        return features

    def _predict_displacements(self, pooled_context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @auto_fp16(apply_to=('driving_output_features',))
    def forward(self, driving_output_features: torch.Tensor, context_mask: torch.Tensor = None, **kwargs):
        driving_output_features = self._match_parameter_dtype(driving_output_features)
        pooled_context = _masked_mean_pool(driving_output_features, context_mask)
        predicted_displacements = self._predict_displacements(pooled_context)
        predicted_path = torch.cumsum(predicted_displacements, dim=1)

        return {
            'predicted_path': predicted_path,
            'predicted_displacements': predicted_displacements,
        }

    def loss(
        self,
        gt_path: torch.Tensor,
        gt_path_valid: torch.Tensor,
        preds_dicts: dict,
        **kwargs,
    ) -> dict:
        predicted_displacements = preds_dicts['predicted_displacements']
        gt_path = gt_path.to(device=predicted_displacements.device, dtype=predicted_displacements.dtype)
        gt_displacements = _path_to_displacements(gt_path)

        waypoint_mask = _gt_valid_mask(gt_path, gt_path_valid).to(dtype=torch.float32)
        loss = F.smooth_l1_loss(
            predicted_displacements.float(),
            gt_displacements.float(),
            beta=self.loss_beta,
            reduction='none',
        )
        masked_loss = loss * waypoint_mask
        denom = (waypoint_mask.sum() * self.path_coord_dim).clamp(min=1.0)
        loss_traj = masked_loss.sum() / denom
        return {'loss_traj': torch.nan_to_num(loss_traj * self.loss_weight)}

    def get_trajectories(self, preds_dicts: dict, **kwargs):
        predicted_path = preds_dicts['predicted_path']
        return [path.detach().cpu().numpy() for path in predicted_path]


@HEADS.register_module()
class MLPFullContextTrajectoryHead(_DirectRegressionTrajectoryHead):
    """Deterministic full-context MLP trajectory regression head."""

    def __init__(
        self,
        hidden_size: int = 2048,
        mlp_dim: int = 1024,
        num_path_points: int = 6,
        path_coord_dim: int = 2,
        dropout: float = 0.0,
        loss_traj: dict = None,
        init_cfg=None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_path_points=num_path_points,
            path_coord_dim=path_coord_dim,
            loss_traj=loss_traj,
            init_cfg=init_cfg,
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mlp_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_path_points * path_coord_dim),
        )

    def _predict_displacements(self, pooled_context: torch.Tensor) -> torch.Tensor:
        displacements = self.regressor(pooled_context)
        return displacements.view(-1, self.num_path_points, self.path_coord_dim)


@HEADS.register_module()
class GRUFullContextTrajectoryHead(_DirectRegressionTrajectoryHead):
    """Deterministic full-context GRU decoder trajectory regression head."""

    def __init__(
        self,
        hidden_size: int = 2048,
        decoder_dim: int = 1024,
        num_layers: int = 2,
        num_path_points: int = 6,
        path_coord_dim: int = 2,
        dropout: float = 0.0,
        loss_traj: dict = None,
        init_cfg=None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_path_points=num_path_points,
            path_coord_dim=path_coord_dim,
            loss_traj=loss_traj,
            init_cfg=init_cfg,
        )
        self.decoder_dim = decoder_dim
        self.num_layers = num_layers

        self.context_norm = nn.LayerNorm(hidden_size)
        self.context_to_hidden = nn.Linear(hidden_size, num_layers * decoder_dim)
        self.waypoint_embeddings = nn.Parameter(torch.randn(1, num_path_points, decoder_dim) * 0.02)
        self.decoder = nn.GRU(
            input_size=decoder_dim,
            hidden_size=decoder_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, path_coord_dim),
        )

    def _predict_displacements(self, pooled_context: torch.Tensor) -> torch.Tensor:
        batch_size = pooled_context.shape[0]
        context = self.context_norm(pooled_context)
        hidden = self.context_to_hidden(context)
        hidden = hidden.view(batch_size, self.num_layers, self.decoder_dim).transpose(0, 1).contiguous()
        decoder_inputs = self.waypoint_embeddings.expand(batch_size, -1, -1)
        decoder_outputs, _ = self.decoder(decoder_inputs, hidden)
        return self.output_head(decoder_outputs)
