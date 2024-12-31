# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Optional

import torch
from torch import Tensor
import numpy as np

from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig, SampleList,
                         OptSampleList)
from mmseg.registry import MODELS
from .encoder_decoder import EncoderDecoder
from ..utils import resize


def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img


@MODELS.register_module()
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 scales=[1],
                 hr_crop_size=None,
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop

    def extract_unscaled_feat(self, inputs: Tensor) -> List[Tensor]:
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, inputs: Tensor):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(inputs)
                else:
                    scaled_img = self.resize(inputs, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(inputs, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def generate_pseudo_label(self, inputs: Tensor,
                              data_samples: OptSampleList):
        batch_img_metas = [
            data_sample.metainfo for data_sample in data_samples
        ]
        self.update_debug_state()
        out = self.encode_decode(inputs, batch_img_metas)
        if self.debug:
            self.debug_output = self.decode_head.debug_output
        return out

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_inputs = self.blur_downup(inputs)
            else:
                scaled_inputs = self.resize(inputs, s)
            if i >= 1 and self.hr_slide_inference:
                mres_feats.append(self.extract_slide_feat(scaled_inputs))
            else:
                mres_feats.append(self.extract_unscaled_feat(scaled_inputs))
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_inputs.detach()
        seg_logits = self.decode_head.predict(mres_feats, batch_img_metas,
                                              self.test_cfg)
        return seg_logits

    def _forward_train_features(self, inputs):
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(inputs)
            else:
                scaled_img = resize(
                    input=inputs,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[
                    f'Img {i} Scale {s}'] = scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        return mres_feats, prob_vis

    def loss(self,
             inputs: Tensor,
             data_samples: SampleList,
             seg_weight=None,
             return_feat=False,
             return_logits=False) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()

        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(inputs)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        loss_decode = self._decode_head_forward_train(mres_feats, data_samples,
                                                      seg_weight,
                                                      return_logits)
        losses.update(loss_decode)
        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
        return losses