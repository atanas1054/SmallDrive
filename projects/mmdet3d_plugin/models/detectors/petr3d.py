# Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from ...datasets.utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from mmdet3d.models import builder
from ..dense_heads.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoTokenizer, GenerationConfig
from ..utils.misc import load_model
from ..utils.positional_encoding import pos2posemb2d
import torch.nn as nn
import os
import json
import mmcv
from projects.mmdet3d_plugin.models.utils.misc import MLN
from mmdet.models.utils.transformer import inverse_sigmoid
import time
@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""
    def __init__(self,
                 save_path='./results_vlm/',
                 use_grid_mask=False,
                 embed_dims=256,
                 LID=True,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 depth_num=64,
                 depth_start = 1,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 map_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 lm_head=None,
                 trajectory_head=None,
                 tokenizer=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 frozen=True,
                 use_lora=False,
                 pretrained=None,
                 export_onnx=False,
                 generate_text=True,
                 eval_tokens_file=None):
        """
        SmallDrive-specific arguments:
            generate_text (bool): If True, autoregressively generate the text
                answer at test time. Set to False for fast, trajectory-only
                inference with the flow-matching trajectory head.
            eval_tokens_file (str, optional): Text file with one nuScenes
                sample token per line. If given, the LLM / trajectory head is
                only run (and results are only saved) for these samples, e.g.
                the Turning-nuScenes subset. The perception branch still runs
                on every frame so that the temporal memory stays consistent.
        """
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.save_path = save_path
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.query_pos = nn.Sequential(
            nn.Linear(396, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims)
        )

        self.ego_pose_pe = MLN(156, export_onnx=export_onnx)

        self.pts_bbox_head.query_pos = self.query_pos
        self.pts_bbox_head.time_embedding = self.time_embedding
        self.pts_bbox_head.ego_pose_pe = self.ego_pose_pe

        if trajectory_head is not None:
            self.trajectory_head = builder.build_head(trajectory_head)

        if map_head is not None:
            self.map_head = builder.build_head(map_head)
            self.map_head.query_pos = self.query_pos
            self.map_head.time_embedding = self.time_embedding
            self.map_head.ego_pose_pe = self.ego_pose_pe

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        model_max_length=2048,
                                        padding_side="right",
                                        use_fast=False,
                                        )
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer = None
        
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if LID:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (self.position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self.position_encoder = nn.Sequential(
                nn.Linear(depth_num*3, embed_dims*4),
                nn.ReLU(),
                nn.Linear(embed_dims*4, embed_dims),
            )
        
        if lm_head is not None:
            self.lm_head = load_model(lm_head, use_lora, frozen)

        self.generate_text = generate_text
        self.eval_tokens = None
        if eval_tokens_file is not None:
            with open(eval_tokens_file, 'r') as f:
                self.eval_tokens = set(line.strip() for line in f if line.strip())

        self.test_flag = False

    @property
    def with_map_head(self):
        """bool: Whether the detector has a map head."""
        return hasattr(self,
                       'map_head') and self.map_head is not None
        
    
    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None
        
    @property
    def with_trajectory_head(self):
        """bool: Whether the detector has a trajectory head."""
        return hasattr(self, 'trajectory_head') and self.trajectory_head is not None

    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()

        img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B), C, H, W)


        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img)
        return img_feats


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def position_embeding(self, data, memory_centers, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = memory_centers.repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)

        return coords_position_embeding

    def _get_leakage_mask(self, input_ids, vlm_labels, vision_embeded, output_length):
        """
        Generates a mask to hide the Ground Truth Answer from the Diffusion Head.
        1 = Valid Context (Image + Prompt)
        0 = Ignored (Answer + Padding)
        """
        B = input_ids.shape[0]
        vis_token_len = vision_embeded.shape[1]
        
        # Initialize mask with 0s
        mask = torch.zeros((B, output_length), device=input_ids.device, dtype=torch.long)
        
        for i in range(B):
            # A. Find where the Answer starts in the RAW text (input_ids)
            # vlm_labels is -100 (IGNORE_INDEX) for Prompt/Image, and valid ID for Answer.
            valid_indices = torch.nonzero(vlm_labels[i] != IGNORE_INDEX).flatten()
            
            if len(valid_indices) > 0:
                raw_answer_start = valid_indices[0].item()
            else:
                # Fallback: If no answer found (shouldn't happen in training), use full length
                raw_answer_start = input_ids.shape[1]

            # B. Calculate the "Shift" caused by Image Insertion
            # We count how many <image> tokens appear BEFORE the answer starts.
            # IMAGE_TOKEN_INDEX is -200
            num_images_before_answer = (input_ids[i, :raw_answer_start] == IMAGE_TOKEN_INDEX).sum().item()
            
            # The shift formula: (Visual_Len - 1 placeholder) * Num_Images
            shift = num_images_before_answer * (vis_token_len - 1)
            
            # C. Determine the Cutoff index in the EXPANDED hidden_states
            expanded_cutoff = raw_answer_start + shift
            
            # Clamp to ensure we don't go out of bounds
            expanded_cutoff = min(expanded_cutoff, output_length)
            
            # D. Set Mask: Keep everything BEFORE the answer
            mask[i, :expanded_cutoff] = 1
            
        return mask
    
    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          gt_path,
                          gt_path_valid,
                          lane_pts,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        B = data['img'].shape[0]
        location = self.prepare_location(img_metas, **data)

        outs_roi = self.forward_roi_head(location, **data)

        pos_embed = self.position_embeding(data, location, img_metas)

        vision_embeded_obj = None
        vision_embeded_map = None

        losses = dict()
        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
            vision_embeded_obj = det_query.clone()

            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses.update(self.pts_bbox_head.loss(*loss_inputs))
            
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            loss_inputs = [lane_pts, outs, img_metas]
            losses.update(self.map_head.loss(*loss_inputs))
                    
        if self.with_lm_head:
            # Visual tokens for the LLM: [scene queries, can_bus/ego token, top-k object tokens]
            # (+ lane queries if a map head is used).
            vision_embeded = []
            if vision_embeded_obj is not None:
                vision_embeded.append(vision_embeded_obj)
            if vision_embeded_map is not None:
                vision_embeded.append(vision_embeded_map)
            if vision_embeded:
                vision_embeded = torch.cat(vision_embeded, dim=1)

            if isinstance(input_ids, list):
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                vlm_labels = torch.nn.utils.rnn.pad_sequence(
                    vlm_labels, batch_first=True, padding_value=IGNORE_INDEX)
                vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)

            # 1. LLM forward (language loss)
            lm_outputs = self.lm_head(
                input_ids=input_ids,
                attention_mask=vlm_attn_mask,
                labels=vlm_labels,
                images=vision_embeded,
                output_hidden_states=True,
                use_cache=False
            )
            losses.update(vlm_loss=lm_outputs[0])

            # 2. Flow-matching trajectory head conditioned on the last decoder layer.
            if self.with_trajectory_head:
                full_llm_context = lm_outputs.hidden_states[-1]

                # Anti-leakage mask: keep only [visual tokens + prompt] and hide the
                # ground-truth answer tokens (1 = valid context, 0 = answer / padding).
                context_mask = self._get_leakage_mask(
                    input_ids=input_ids,
                    vlm_labels=vlm_labels,
                    vision_embeded=vision_embeded,
                    output_length=full_llm_context.shape[1]
                )
                mask_expanded = context_mask.unsqueeze(-1).to(full_llm_context.dtype)
                clean_context = full_llm_context * mask_expanded

                preds_traj = self.trajectory_head.forward(
                    driving_output_features=clean_context,
                    context_mask=context_mask
                )
                traj_loss = self.trajectory_head.loss(gt_path, gt_path_valid, preds_traj)
                losses.update(traj_loss)

        if self.with_img_roi_head:
            loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
            losses2d = self.img_roi_head.loss(*loss2d_inputs)
            losses.update(losses2d) 

        return losses
    

    # @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      input_ids=None,
                      vlm_labels=None,
                      gt_path=None,
                      gt_path_valid=None,
                      lane_pts=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            vlm_labels = vlm_labels[:, :self.tokenizer.model_max_length]
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None

        data['img_feats'] = self.extract_feat(data['img'])

        losses = self.forward_pts_train(gt_bboxes_3d,
                                    gt_labels_3d, gt_bboxes,
                                    gt_labels, img_metas, centers2d, 
                                    depths, input_ids, vlm_labels, vlm_attn_mask, gt_path,
                                    gt_path_valid, lane_pts, **data)

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        if not self.test_flag: #for interval evaluation
            if self.with_pts_bbox:
                self.pts_bbox_head.reset_memory()
            if self.with_map_head:
                self.map_head.reset_memory()
            self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key not in ['img', 'input_ids']:
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        pos_embed = self.position_embeding(data, location, img_metas)
        bbox_results = []
        trajectory_results = []
        vision_embeded_obj = None
        vision_embeded_map = None

        if self.with_pts_bbox:
            outs, det_query = self.pts_bbox_head(img_metas, pos_embed, **data)
            vision_embeded_obj = det_query.clone()
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas)
            for bboxes, scores, labels in bbox_list:
                bbox_results.append(bbox3d2result(bboxes, scores, labels))

        lane_results = []
        if self.with_map_head:
            outs, map_query = self.map_head(img_metas, pos_embed, **data)
            vision_embeded_map = map_query.clone()
            lane_results = self.map_head.get_bboxes(outs, img_metas)

        generated_text = []
        sample_idx = img_metas[0]['sample_idx']
        # Optionally restrict the language/trajectory part to a subset of samples
        # (e.g. Turning-nuScenes). Perception above always runs to keep the memory bank in sync.
        run_llm = self.with_lm_head and (self.eval_tokens is None or sample_idx in self.eval_tokens)

        if run_llm:
            # In server mode (e.g. latency profiling) nothing is written to disk.
            is_server_mode = getattr(self, 'server_mode', False)
            if not is_server_mode:
                mmcv.mkdir_or_exist(self.save_path)

            vision_embeded = []
            if vision_embeded_obj is not None:
                vision_embeded.append(vision_embeded_obj)
            if vision_embeded_map is not None:
                vision_embeded.append(vision_embeded_map)
            if vision_embeded:
                vision_embeded = torch.cat(vision_embeded, dim=1)

            for i, input_ids in enumerate(data['input_ids'][0]):
                input_ids = input_ids.unsqueeze(0)

                if self.with_trajectory_head:
                    # 1. Single LLM prefill pass over [visual tokens + prompt].
                    (
                        _, _, _, _, inputs_embeds, _
                    ) = self.lm_head.prepare_inputs_labels_for_multimodal(
                        input_ids, None, None, None, None, vision_embeded, None
                    )
                    outputs = self.lm_head.model(
                        inputs_embeds=inputs_embeds,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    full_llm_context = outputs.hidden_states[-1]

                    # 2. Context mask (1 = prompt + expanded visual tokens, 0 = padding).
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    num_valid_text = input_ids.ne(pad_id).sum().item()
                    if vision_embeded is not None:
                        num_images = (input_ids == IMAGE_TOKEN_INDEX).sum().item()
                        expansion_shift = num_images * (vision_embeded.shape[1] - 1)
                    else:
                        expansion_shift = 0
                    seq_len_out = full_llm_context.shape[1]
                    inference_mask = torch.zeros((1, seq_len_out), device=input_ids.device, dtype=torch.long)
                    inference_mask[:, :min(num_valid_text + expansion_shift, seq_len_out)] = 1

                    # 3. Flow-matching trajectory generation (Euler integration).
                    preds_traj = self.trajectory_head.forward(
                        driving_output_features=full_llm_context,
                        context_mask=inference_mask
                    )
                    trajectory_arrays = self.trajectory_head.get_trajectories(preds_traj)
                    trajectory_results = [path.tolist() for path in trajectory_arrays]

                # 4. Optional autoregressive text answer (explanations, QA, or text waypoints
                #    for the autoregressive baseline).
                answer = ""
                if self.generate_text:
                    output_ids = self.lm_head.generate(
                        inputs=input_ids,
                        images=vision_embeded,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.75,
                        num_beams=1,
                        max_new_tokens=320,
                        use_cache=True
                    )
                    answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

                if not is_server_mode:
                    generated_text.append(
                        dict(
                            Q=img_metas[0]['vlm_labels'].data[i],
                            A=answer,
                            trajectory=trajectory_results
                        ))

            if not is_server_mode:
                with open(self.save_path + sample_idx, 'w') as file:
                    json.dump(generated_text, file)

        return bbox_results, generated_text, lane_results, trajectory_results

    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'])
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts, generated_text, lane_results, trajectory_results = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        bbox_list[0]['text_out'] = generated_text
        bbox_list[0]['lane_results'] = lane_results
        bbox_list[0]['trajectory_results'] = trajectory_results
        return bbox_list

    
