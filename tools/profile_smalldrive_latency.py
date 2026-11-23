import argparse
import importlib
import os
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass


class CudaTimer:
    def __init__(self):
        self.times = defaultdict(list)
        self.enabled = False
        self.llm_forward_calls = 0
        self.additive_keys = {'q_former'}
        self._sample_accum = defaultdict(float)

    def start_sample(self):
        self.llm_forward_calls = 0
        self._sample_accum = defaultdict(float)

    def finish_sample(self):
        if not self.enabled:
            return
        for name, elapsed_ms in self._sample_accum.items():
            self.times[name].append(elapsed_ms)

    @contextmanager
    def record(self, name):
        if not self.enabled:
            yield
            return

        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if name in self.additive_keys:
            self._sample_accum[name] += elapsed_ms
        else:
            self.times[name].append(elapsed_ms)

    def mean(self, name):
        values = self.times.get(name, [])
        return statistics.mean(values) if values else 0.0

    def std(self, name):
        values = self.times.get(name, [])
        return statistics.pstdev(values) if len(values) > 1 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile SmallDrive component latency on validation samples.')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', type=int, default=100,
                        help='number of timed samples')
    parser.add_argument('--warmup', type=int, default=10,
                        help='number of warmup samples skipped from averages')
    parser.add_argument('--workers-per-gpu', type=int, default=2,
                        help='dataloader workers')
    parser.add_argument('--data-root', default=None,
                        help='optional nuScenes root; rewrites ./data/nuscenes/ paths')
    parser.add_argument('--fuse-conv-bn', action='store_true',
                        help='fuse conv and bn before profiling')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override config options, e.g. model.save_path=...')
    parser.add_argument('--show-std', action='store_true',
                        help='also print per-component population std')
    return parser.parse_args()


def import_plugin(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return

    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        module_parts = os.path.dirname(plugin_dir).split('/')
    else:
        module_parts = os.path.dirname(config_path).split('/')

    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path = module_path + '.' + part
    importlib.import_module(module_path)


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.data_root is not None:
        rewrite_data_paths(cfg, args.data_root)

    import_plugin(cfg, args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    else:
        samples_per_gpu = 1
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = args.workers_per_gpu
    return cfg


def rewrite_data_paths(cfg, data_root):
    data_root = data_root.rstrip('/') + '/'
    old_prefixes = ('./data/nuscenes/', 'data/nuscenes/')

    def rewrite_value(value):
        if isinstance(value, str):
            for old_prefix in old_prefixes:
                if value.startswith(old_prefix):
                    return data_root + value[len(old_prefix):]
        if isinstance(value, list):
            return [rewrite_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(rewrite_value(item) for item in value)
        if isinstance(value, dict):
            for key in list(value.keys()):
                value[key] = rewrite_value(value[key])
            return value
        return value

    cfg_dict = cfg._cfg_dict if isinstance(cfg, Config) else cfg
    rewrite_value(cfg_dict)

    data_cfg = cfg_dict.get('data', {})
    for split in ('train', 'val', 'test'):
        split_cfg = data_cfg.get(split, None)
        if isinstance(split_cfg, dict) and split_cfg.get('type') == 'CustomNuScenesDataset':
            split_cfg.setdefault('lane_path', data_root + 'data_dict_sample.pkl')
            split_cfg.setdefault('lane_anno_file', data_root + 'data_dict_subset_B_val.pkl')


def build_data_loader(cfg, data_root=None):
    dataset = build_dataset(cfg.data.test)
    if data_root is not None:
        rewrite_dataset_file_paths(dataset, data_root)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', None),
    )
    return data_loader


def rewrite_dataset_file_paths(dataset, data_root):
    data_root = data_root.rstrip('/') + '/'
    old_prefixes = ('./data/nuscenes/', 'data/nuscenes/')

    def rewrite_value(value):
        if isinstance(value, str):
            for old_prefix in old_prefixes:
                if value.startswith(old_prefix):
                    return data_root + value[len(old_prefix):]
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = rewrite_value(item)
            return value
        if isinstance(value, tuple):
            return tuple(rewrite_value(item) for item in value)
        if isinstance(value, dict):
            for key in list(value.keys()):
                value[key] = rewrite_value(value[key])
            return value
        return value

    if hasattr(dataset, 'data_infos'):
        rewrite_value(dataset.data_infos)


def patch_component_timers(model, timer):
    module = model.module

    original_extract_img_feat = module.extract_img_feat

    def timed_extract_img_feat(*args, **kwargs):
        with timer.record('vision_encoder'):
            return original_extract_img_feat(*args, **kwargs)

    module.extract_img_feat = timed_extract_img_feat

    if hasattr(module, 'pts_bbox_head') and module.pts_bbox_head is not None:
        original_qformer_forward = module.pts_bbox_head.forward

        def timed_qformer_forward(*args, **kwargs):
            with timer.record('q_former'):
                return original_qformer_forward(*args, **kwargs)

        module.pts_bbox_head.forward = timed_qformer_forward

    if hasattr(module, 'map_head') and module.map_head is not None:
        original_map_forward = module.map_head.forward

        def timed_map_forward(*args, **kwargs):
            with timer.record('q_former'):
                return original_map_forward(*args, **kwargs)

        module.map_head.forward = timed_map_forward

    uses_trajectory_head = hasattr(module, 'trajectory_head') and module.trajectory_head is not None

    if hasattr(module, 'lm_head') and module.lm_head is not None:
        original_llm_forward = module.lm_head.model.forward

        def timed_llm_forward(*args, **kwargs):
            name = 'llm_prefill'
            if not uses_trajectory_head:
                timer.llm_forward_calls += 1
                name = 'llm_prefill' if timer.llm_forward_calls == 1 else 'llm_decode_step'
            with timer.record(name):
                return original_llm_forward(*args, **kwargs)

        module.lm_head.model.forward = timed_llm_forward

        if not uses_trajectory_head:
            original_generate = module.lm_head.generate

            def timed_generate(*args, **kwargs):
                with timer.record('trajectory_generation'):
                    return original_generate(*args, **kwargs)

            module.lm_head.generate = timed_generate

    if uses_trajectory_head:
        original_flow_forward = module.trajectory_head.forward

        def timed_flow_forward(*args, **kwargs):
            with timer.record('trajectory_generation'):
                return original_flow_forward(*args, **kwargs)

        module.trajectory_head.forward = timed_flow_forward


def build_profile_model(cfg, args, dataset):
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    # Avoid file I/O in Petr3D.simple_test_pts while keeping the same model path.
    model.module.server_mode = True
    # With a trajectory head, the trajectory only needs the LLM prefill: skip
    # autoregressive text generation (this is the latency reported in the paper).
    if getattr(model.module, 'trajectory_head', None) is not None:
        model.module.generate_text = False
    return model


def format_ms(value, show_std=False, std=0.0):
    if show_std:
        return f'{value:.2f} ± {std:.2f}'
    return f'{value:.2f}'


def print_table(timer, show_std):
    rows = [
        ('Vision encoder', 'vision_encoder'),
        ('Q-Former', 'q_former'),
        ('LLM prefill', 'llm_prefill'),
        ('Trajectory generation', 'trajectory_generation'),
        ('Total', 'total'),
    ]

    print('\n| Component | Time (ms) |')
    print('|---|---:|')
    for label, key in rows:
        mean_ms = timer.mean(key)
        std_ms = timer.std(key)
        print(f'| {label} | {format_ms(mean_ms, show_std, std_ms)} |')


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    data_loader = build_data_loader(cfg, args.data_root)
    model = build_profile_model(cfg, args, data_loader.dataset)

    timer = CudaTimer()
    patch_component_timers(model, timer)

    total_needed = args.warmup + args.samples
    print(
        f'Profiling {args.samples} samples after {args.warmup} warmup samples '
        f'on {torch.cuda.get_device_name(0)}.'
    )

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            timer.enabled = i >= args.warmup
            timer.start_sample()
            with timer.record('total'):
                model(return_loss=False, rescale=True, **data)
            timer.finish_sample()

            if i + 1 >= total_needed:
                break

    if len(timer.times['total']) == 0:
        raise RuntimeError('No timed samples were collected. Reduce --warmup.')

    print_table(timer, args.show_std)
    print(
        '\nNote: Total is the full model forward latency excluding dataloader time. '
        'Component rows are timed at the main module boundaries and may not sum '
        'exactly to Total because input preparation, formatting, masks, and small '
        'bookkeeping operations are outside those four component timers.'
    )


if __name__ == '__main__':
    main()
