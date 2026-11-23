# SmallDrive (GCPR 2026) main configuration.
#   - Vision encoder : EVA-02-B, 640x640 input
#   - Q-Former3D     : StreamPETR-based, 256 global scene queries + top-50 object tokens
#                      (no lane/map queries -> 256 + 1 ego/command + 50 = 307 visual tokens)
#   - LLM            : TinyLlama 1.1B, LoRA (rank 128)
#   - Trajectory     : conditional flow-matching transformer (8 layers, 8 heads, dim 1024),
#                      conditioned on the last decoder layer of the LLM, 10 Euler steps
#   - Schedule       : 24 epochs, 4 GPUs x batch size 4
_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# NOTE: the schedule is defined in iterations. If you train with a different number of
# GPUs / batch size, update these two values so that num_iters_per_epoch stays correct.
num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 24
llm_path = 'ckpts/omnidrive_tiny_pretrain/finetune-1.1b-llava-tiny-llama-eva640/'

collect_keys_test = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus']
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus', 'gt_path', 'gt_path_valid']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Petr3D',
    save_path='./results/smalldrive/',  # where per-sample predictions (json) are written at test time
    generate_text=True,      # set False for fast trajectory-only inference (planning eval / latency)
    eval_tokens_file=None,   # e.g. 'evaluation/turning_nuscenes_val_tokens.txt' for Turning-nuScenes
    use_grid_mask=True,
    frozen=False,
    use_lora=True,
    tokenizer=llm_path,
    lm_head=llm_path,  # set to None if don't use llm head
    img_backbone=dict(
        type='EVAViT',
        img_size=640,
        patch_size=16,
        window_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4*2/3,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        qkv_bias=True,
        drop_path_rate=0.1,
        flash_attn=True,
        with_cp=True,
        frozen=False),
    map_head=None,  # SmallDrive drops OmniDrive's 256 lane queries
    pts_bbox_head=dict(
        type='StreamPETRHead',
        num_classes=10,
        in_channels=768,
        out_dims=2048,
        num_query=600,
        with_mask=True,
        memory_len=600,
        topk_proposals=300,
        num_propagated=300,
        num_extra=256,                       # global scene queries
        num_topk_objects=50,                 # object tokens passed to the LLM (0 = disable)
        obj_token_content='feat_box_score',  # 'feat_box_score' | 'feat_only' | 'geo_only'
        n_control=11,  # align with centerline query defination
        match_with_velo=False,
        scalar=10,  # noise groups
        noise_scale=1.0,
        dn_weight=1.0,  # dn loss weight
        split=0.75,  # positive rate
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='PETRTemporalTransformer',
            input_dimension=256,
            output_dimension=256,
            num_layers=6,
            embed_dims=256,
            num_heads=8,
            feedforward_dims=2048,
            dropout=0.1,
            with_cp=True,
            flash_attn=True,
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),),
    # Flow-matching trajectory expert
    trajectory_head=dict(
        type='DiffusionTransformerTrajectoryHead',
        hidden_size=2048,       # must match the TinyLlama hidden size
        transformer_dim=1024,   # 0.5 x LLM hidden size
        num_layers=8,
        num_heads=8,
        num_inference_steps=10,  # Euler steps at inference
        flow_sig_min=0.001,
        flow_sampling="beta",    # timestep sampling during training
        flow_alpha=1.5,
        flow_beta=1.0,
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),))
)


dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')


ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
         with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQA',
         base_vqa_path='./data/nuscenes/vqa/train/',
         base_desc_path='./data/nuscenes/desc/train/',
         base_conv_path='./data/nuscenes/conv/train/',
         base_key_path='./data/nuscenes/keywords/train/',
         tokenizer=llm_path,
         max_length=2048,
         ignore_type=[],
         lane_objs_info="./data/nuscenes/lane_obj_train.pkl"),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['lane_pts', 'input_ids', 'vlm_labels', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadAnnoatationVQATest',
         base_vqa_path='./data/nuscenes/vqa/val/',
         base_conv_path='./data/nuscenes/conv/val/',
         base_counter_path='./data/nuscenes/eval_cf/',
         # "planning" -> open-loop planning prompt; see smalldrive_lang.py for the QA evaluation.
         # Please don't test all question types in a single run, it takes a long time.
         load_type=["planning"],
         tokenizer=llm_path,
         max_length=2048,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys_test,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['input_ids', 'img'] + collect_keys_test,
                 meta_keys=('sample_idx', 'vlm_labels', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_train.pkl',
        seq_split_num=1,  # streaming video training
        seq_mode=True,  # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        warmup_split_num=10,  # lane det and vlm need short term temporal fusion in the early stage of training
        num_iters_to_seq=num_iters_per_epoch,
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# Base LR 1e-4; the trajectory expert uses lr_mult=0.1 (i.e. 1e-5).
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
    paramwise_cfg={
        'decay_rate': 0.9,
        'head_decay_rate': 4.0,
        'lm_head_decay_rate': 1.0,
        'decay_type': 'vit_wise',
        'num_layers': 24,
        'custom_keys': {
            'trajectory_head': dict(lr_mult=0.1),
            'lm_head': dict(lr_mult=4.0),
        }
    }
)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)

find_unused_parameters = False  # when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch//2, max_keep_ckpts=3)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = "ckpts/omnidrive_tiny_pretrain/eva_base_remapped.pth"
resume_from = None
