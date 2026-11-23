import importlib.util
from pathlib import Path

import torch
from mmdet.models import HEADS, build_head


def _load_head_module():
    if all(
        name in HEADS.module_dict
        for name in ['MLPFullContextTrajectoryHead', 'GRUFullContextTrajectoryHead']
    ):
        return

    module_path = (
        Path(__file__).resolve().parents[1]
        / 'projects'
        / 'mmdet3d_plugin'
        / 'models'
        / 'dense_heads'
        / 'direct_regression_trajectory_head.py'
    )
    spec = importlib.util.spec_from_file_location('direct_regression_trajectory_head', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


_load_head_module()


def _build_head(head_type):
    cfg = dict(
        type=head_type,
        hidden_size=8,
        num_path_points=6,
        path_coord_dim=2,
        loss_traj=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
    )
    if head_type == 'MLPFullContextTrajectoryHead':
        cfg['mlp_dim'] = 16
    else:
        cfg['decoder_dim'] = 16
        cfg['num_layers'] = 2
    return build_head(cfg)


def test_direct_regression_heads_forward_shapes_and_extraction():
    features = torch.randn(3, 5, 8)
    context_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    for head_type in ['MLPFullContextTrajectoryHead', 'GRUFullContextTrajectoryHead']:
        head = _build_head(head_type)
        preds = head(driving_output_features=features, context_mask=context_mask)

        assert preds['predicted_displacements'].shape == (3, 6, 2)
        assert preds['predicted_path'].shape == (3, 6, 2)
        assert torch.allclose(preds['predicted_path'], preds['predicted_displacements'].cumsum(dim=1))
        assert len(head.get_trajectories(preds)) == 3


def test_direct_regression_loss_respects_waypoint_valid_mask():
    head = _build_head('MLPFullContextTrajectoryHead')
    gt_path = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 1.0], [6.0, 1.0], [10.0, 1.0], [15.0, 1.0], [21.0, 1.0]],
            [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0], [12.0, 0.0]],
        ]
    )
    gt_displacements = torch.zeros_like(gt_path)
    gt_displacements[:, 0, :] = gt_path[:, 0, :]
    gt_displacements[:, 1:, :] = gt_path[:, 1:, :] - gt_path[:, :-1, :]

    predicted_displacements = gt_displacements.clone()
    predicted_displacements[:, 2:, :] = predicted_displacements[:, 2:, :] + 100.0
    gt_path_valid = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    loss = head.loss(
        gt_path=gt_path,
        gt_path_valid=gt_path_valid,
        preds_dicts={'predicted_displacements': predicted_displacements},
    )['loss_traj']

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_direct_regression_loss_handles_zero_valid_mask():
    head = _build_head('GRUFullContextTrajectoryHead')
    gt_path = torch.randn(2, 6, 2)
    predicted_displacements = torch.randn(2, 6, 2)
    gt_path_valid = torch.zeros(2, 6, dtype=torch.bool)

    loss = head.loss(
        gt_path=gt_path,
        gt_path_valid=gt_path_valid,
        preds_dicts={'predicted_displacements': predicted_displacements},
    )['loss_traj']

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
