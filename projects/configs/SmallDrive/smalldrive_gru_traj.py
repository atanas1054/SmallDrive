# Ablation: replace the flow-matching trajectory expert with a deterministic
# direct-regression GRU decoder (pooled LLM context -> 6 sequential waypoint displacements).
_base_ = ['./smalldrive.py']

model = dict(
    save_path='./results/smalldrive_gru_traj/',
    trajectory_head=dict(
        _delete_=True,
        type='GRUFullContextTrajectoryHead',
        hidden_size=2048,
        decoder_dim=1024,
        num_layers=2,
        num_path_points=6,
        path_coord_dim=2,
        dropout=0.0,
        loss_traj=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
    ),
)
