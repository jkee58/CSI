# Copyright (c) OpenMMLab. All rights reserved.
# Modifications:
# - Support batched slide
# - Update debug output system

import logging
from typing import List, Optional

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from ..utils.visualization import prepare_debug_out, subplotimg
from .base import BaseSegmentor
from ..utils import resize


@MODELS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

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
                 text_embeddings_path=None,
                 extract_text_feat=False,
                 renorm_clip_img=False):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # MaskCLIPViT
        self.text_embeddings_path = text_embeddings_path
        if self.text_embeddings_path is not None:
            self.text_embeddings = np.load(self.text_embeddings_path)
        self.extract_text_feat = extract_text_feat
        assert (not self.extract_text_feat
                or (self.extract_text_feat and
                    (self.text_embeddings_path is not None)))
        self.renorm_clip_img = renorm_clip_img
        if self.renorm_clip_img:
            print_log('Renormalize clip image', logger="current")

        self.automatic_debug = True
        self.debug = False
        self.debug_output = {}

        self.local_iter = 0

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    # def extract_feat(self, inputs: Tensor) -> List[Tensor]:
    #     """Extract features from images."""
    #     x = self.backbone(inputs)
    #     if self.with_neck:
    #         x = self.neck(x)
    #     return x

    def extract_feat(self, inputs: Tensor):
        """Extract features from images."""

        if self.renorm_clip_img:
            inputs = self.renormalize_img_for_clip(inputs)

        if self.extract_text_feat:
            visual_feat = self.backbone(inputs)
            text_feat = self.text_embeddings
            text_feat = torch.from_numpy(text_feat).to(inputs.device)
            return [visual_feat, text_feat]
        else:
            x = self.backbone(inputs)
            if self.with_neck:
                x = self.neck(x)

            if self.test_cfg.get('save_feats', False):
                self.feats = x[-1]
            return x

    def generate_pseudo_label(self,
                              inputs: Tensor,
                              data_samples: OptSampleList,
                              clip_guided=False):
        self.update_debug_state()
        if self.debug:
            self.debug_output = {
                'Image': inputs,
            }

        if clip_guided:
            out = self.encode_decode(inputs, data_samples, clip_guided=True)
        else:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
            out = self.encode_decode(inputs, batch_img_metas)
            if self.debug:
                self.debug_output.update(self.decode_head.debug_output)
                self.debug_output['Pred'] = out.cpu().numpy()
        return out

    def encode_decode(self,
                      inputs: Tensor,
                      batch_img_metas: List[dict],
                      clip_guided=False) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)

        if clip_guided:
            seg_logits = self.decode_head.clip_guided_predict(
                x, batch_img_metas, self.test_cfg, inputs)
        else:
            seg_logits = self.decode_head.predict(x, batch_img_metas,
                                                  self.test_cfg)
        return seg_logits

    def _decode_head_forward_train(self,
                                   inputs: List[Tensor],
                                   data_samples: SampleList,
                                   seg_weight=None,
                                   return_logits=False,
                                   img=None) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(
            inputs, data_samples, self.train_cfg, seg_weight=seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self,
                                      inputs: List[Tensor],
                                      data_samples: SampleList,
                                      seg_weight=None) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg,
                                         seg_weight)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def update_debug_state(self):
        self.debug_output = {}
        self.decode_head.debug = self.debug
        if self.with_auxiliary_head:
            self.auxiliary_head.debug = self.debug

    def loss(self,
             inputs: Tensor,
             data_samples: SampleList,
             seg_weight=None,
             return_feat=False,
             return_logits=False,
             return_fused=False,
             loss_name=None,
             **kwargs) -> dict:
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

        x = self.extract_feat(inputs)

        losses = dict()
        if return_feat:
            if self.extract_text_feat:
                losses['features'] = x[0][0]
            else:
                losses['features'] = x

        loss_decode = self._decode_head_forward_train(x, data_samples,
                                                      seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        if self.debug:
            self.process_debug(inputs)

        return losses

    def process_debug(self, img):
        self.debug_output = {
            'Image': img,
            **self.decode_head.debug_output,
        }
        if self.with_auxiliary_head:
            self.debug_output.update(
                add_prefix(self.auxiliary_head.debug_output, 'Aux'))
        if self.automatic_debug:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'encdec_debug')
            os.makedirs(out_dir, exist_ok=True)
            means, stds = self.data_preprocessor.mean, self.data_preprocessor.std
            for j in range(img.shape[0]):
                rows, cols = 1, len(self.debug_output)
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.92,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                for k, (n, v) in enumerate(self.debug_output.items()):
                    subplotimg(axs[k],
                               **prepare_debug_out(n, v[j], means, stds))
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
            del self.debug_output

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        # # Use CRF for prediction
        # if 'dense_crf' in self.test_cfg.keys() and self.test_cfg['dense_crf']:
        #     for b in range(inputs.shape[0]):
        #         seg_logits[b] = torch.from_numpy(
        #             dense_crf(inputs[b].cpu(), torch.log(seg_logits[b])))

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        _, _, seg_logits = self.decode_head.forward(x, return_fused=False)
        return seg_logits

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = inputs[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            batch_img_metas[0]['img_shape'] = crop_imgs.shape[2:]
            crop_seg_logits = self.encode_decode(
                crop_imgs, batch_img_metas)  # use only target decoder
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = inputs[:, :, y1:y2, x1:x2]
                    # change the image shape to patch shape
                    batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                    # the output of encode_decode is seg logits tensor map
                    # with shape [N, C, H, W]
                    crop_seg_logit = self.encode_decode(
                        crop_img, batch_img_metas)
                    preds += F.pad(crop_seg_logit,
                                   (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)
        return seg_logits

    def inference(self,
                  inputs: Tensor,
                  batch_img_metas: List[dict],
                  return_feat=False) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)
        if hasattr(self.decode_head, 'debug_output_attention'
                   ) and self.decode_head.debug_output_attention:
            output = seg_logit
        else:
            output = F.softmax(seg_logit, dim=1)

        return output

    def aug_test(self, imgs, img_metas, rescale=True):  # => Convert to TTA
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0])
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i])
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def renormalize_img_for_clip(self, img):
        loader_mean, loader_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        clip_mean, clip_std = [0.48145466, 0.4578275, 0.40821073
                               ], [0.26862954, 0.26130258, 0.27577711]
        loader_mean = torch.tensor(
            loader_mean, device=img.device).view(1, -1, 1, 1)
        loader_std = torch.tensor(
            loader_std, device=img.device).view(1, -1, 1, 1)
        clip_mean = torch.tensor(
            clip_mean, device=img.device).view(1, -1, 1, 1)
        clip_std = torch.tensor(clip_std, device=img.device).view(1, -1, 1, 1)
        return (img * loader_std + loader_mean - clip_mean) / clip_std

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(
            f'Loaded text embeddings from {self.text_embeddings_path}',
            logger="current")
