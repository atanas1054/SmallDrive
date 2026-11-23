# Environment Setup

SmallDrive is built on top of [OmniDrive](https://github.com/NVlabs/OmniDrive) and uses the same environment and data.

**1. Install packages**

```shell
cd /path/to/SmallDrive
conda create -n smalldrive python=3.9 -y
conda activate smalldrive
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install flash-attn==2.5.6
pip install transformers==4.31.0
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6
pip install -e .
cd ..
git clone https://github.com/OpenDriveLab/OpenLane-V2.git
cd OpenLane-V2
pip install -e .
cd ..
pip install -r requirements.txt
```

`mmdetection3d/` and `OpenLane-V2/` must live inside the repository root (the configs reference `mmdetection3d/configs/_base_`).

Optional, for language evaluation only: `pip install pycocoevalcap bert-score` and
`pip install git+https://github.com/bckim92/language-evaluation.git`.

**2. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

**3. Download the OmniDrive-nuScenes info files and QA annotations**

Download the [info files](https://github.com/NVlabs/OmniDrive/releases/download/v1.0/data_nusc.zip), unzip them and move the files into `data/nuscenes/`.

The pkl info files are generated with StreamPETR's data converter. The GT is converted from the lidar to the ego coordinate system with `update_coords.py` (to align with OpenLane-V2) and canbus/command information is added.

**4. Download pretrained weights**

SmallDrive is initialized from the 2D-pretrained OmniDrive TinyLlama checkpoint (EVA-02-B vision encoder + Q-Former + TinyLlama 1.1B, pretrained on LLaVA data):

```shell
mkdir -p ckpts && cd ckpts
git lfs install
git clone https://huggingface.co/exiawsh/omnidrive_tiny_pretrain
```

Only `eva_base_remapped.pth` and `finetune-1.1b-llava-tiny-llama-eva640/` are needed.

**5. Folder structure**

```
SmallDrive
├── projects/
│   ├── configs/SmallDrive/
│   └── mmdet3d_plugin/
├── mmdetection3d/
├── OpenLane-V2/
├── tools/
├── evaluation/
├── ckpts/
│   └── omnidrive_tiny_pretrain/
│       ├── eva_base_remapped.pth
│       └── finetune-1.1b-llava-tiny-llama-eva640/
├── data/
│   └── nuscenes/
│       ├── maps/
│       ├── samples/
│       ├── sweeps/
│       ├── v1.0-test/
│       ├── v1.0-trainval/
│       ├── conv/
│       ├── desc/
│       ├── keywords/
│       ├── vqa/
│       ├── eval_cf/
│       ├── nuscenes2d_ego_temporal_infos_train.pkl
│       ├── nuscenes2d_ego_temporal_infos_val.pkl
│       ├── data_dict_sample.pkl
│       ├── data_dict_subset_B.json
│       ├── data_dict_subset_B_val.pkl
│       └── lane_obj_train.pkl
```
