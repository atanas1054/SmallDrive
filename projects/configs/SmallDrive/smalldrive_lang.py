# Language evaluation on OmniDrive-nuScenes (scene description, action explanation,
# counterfactual reasoning, ...). Same model as smalldrive.py, only the test prompts differ.
_base_ = ['./smalldrive.py']

llm_path = 'ckpts/omnidrive_tiny_pretrain/finetune-1.1b-llava-tiny-llama-eva640/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
collect_keys_test = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command', 'can_bus']
ida_aug_conf = {
    "resize_lim": (0.37, 0.45),
    "final_dim": (320, 640),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

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
         load_type=["conv"],
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

model = dict(save_path='./results/smalldrive_lang/', generate_text=True)
data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
