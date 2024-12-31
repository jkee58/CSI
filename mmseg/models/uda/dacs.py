# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

from typing import Dict
import math
import random
from copy import deepcopy
from overrides import overrides

import mmengine
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.utils import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio

import torch
from torch import Tensor
from mmseg.utils import SampleList
from mmseg.visualization import SegLocalVisualizer

from mmengine import MessageHub
from mmengine.logging import print_log
import logging

from mmseg.models.utils import patch_master
from mmseg.models.utils.relabeling_map import RelabelingMap
from collections import Counter
from mmseg.models.utils import resize


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None

        # RUDA Configuration
        self.relabel_with_mapping_info = False
        self.relabel_with_mapping_info_start, self.relabel_with_mapping_info_end = cfg[
            'range_of_relabel_with_mapping_info']

        self.collect_mapping_info = False
        self.collect_mapping_info_start, self.collect_mapping_info_end = cfg[
            'range_of_collect_mapping_info']
        self.enable_patching = self.relabel_with_mapping_info_start > -1
        if self.enable_patching:
            self.patch_bank_capacity: int = cfg['patch_bank_capacity']
            self.checkpoint_for_object_detection: str = cfg['ckpt_for_obj_det']
            self.patcher = patch_master.Patcher(
                mean=self.data_preprocessor.mean.expand([3, 1, 1]),
                std=self.data_preprocessor.std.expand([3, 1, 1]),
                ignore_top=self.psweight_ignore_top,
                ignore_bottom=-self.psweight_ignore_bottom,
                checkpoint_for_object_detection=self.
                checkpoint_for_object_detection)
            self.bank_group = [
                patch_master.PatchBank(
                    patch_label=class_index,
                    patch_capacity=self.patch_bank_capacity)
                for class_index in range(0, 19)
            ]
            self.relabeling_map = RelabelingMap(
                relabeling_cfg=cfg['relabeling_cfg'],
                class_index_to_category=self.patcher.clip_guide.
                class_index_to_category)
            self.total_relabled_count = [0 for _ in range(0, 19)]

        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None
        self.debug_img = False

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        # Visualize
        self.seg_visualizer = SegLocalVisualizer.get_current_instance()

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def _stack_batch_valid(self, batch_data_samples: SampleList) -> Tensor:
        valid_pseudo_masks = [
            data_sample.valid_pseudo_mask for data_sample in batch_data_samples
        ]
        return torch.stack(valid_pseudo_masks, dim=0)

    def _run_forward(
            self,
            data,  ## Override base_model.py
            mode: str,
            return_feat=False):
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        self.local_iter = MessageHub.get_current_instance().get_info(
            key='iter')

        if isinstance(data, dict):
            results = self(**data, mode=mode, return_feat=return_feat)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode, return_feat=return_feat)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    @overrides
    def train_step(self, data, optim_wrapper) -> Dict[str, torch.Tensor]:
        """The iteration step during non-distributed training.
        
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        src_data, trg_data = data['src_data'], data['trg_data']
        data['trg_data'] = self.data_preprocessor(trg_data, training=True)
        data['src_data'] = self.data_preprocessor(src_data, training=True)

        optim_wrapper.zero_grad()
        log_vars = self._run_forward(data, mode='loss')  # type: ignore
        optim_wrapper.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        return log_vars

    def val_step(self, data) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, training=False)
        return self._run_forward({'trg_data': data},
                                 mode='predict')  # type: ignore

    def test_step(self, data) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, training=False)
        pred = self._run_forward({'trg_data': data}, mode='predict')
        return pred

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self,
                       img,
                       batch_data_samples,
                       feat=None,
                       fdist_mask=None):
        if batch_data_samples is not None:
            gt = self._stack_batch_gt(batch_data_samples)
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                if self.get_imnet_model().extract_text_feat:
                    feat_imnet = feat_imnet[0][0]
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                if fdist_mask is None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()

                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    self.debug_gt_rescale = gt_rescaled
                else:
                    _, _, H, W = feat[lay].shape
                    fdist_mask = resize(fdist_mask.float(), size=(H, W)).bool()

                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)

                self.debug_fdist_mask = fdist_mask
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self.parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self, debug):
        # debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def loss(self,
             src_data,
             trg_data,
             rare_class=None,
             valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = src_data['inputs'].shape[0]
        dev = src_data['inputs'].device
        self.custom_debug = dict()

        if 'valid_pseudo_mask' in vars(trg_data['data_samples'][0]).keys():
            valid_pseudo_mask = self._stack_batch_valid(
                trg_data['data_samples']).unsqueeze(1).cuda()
        else:
            valid_pseudo_mask = None

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        seg_debug = {}

        means = self.data_preprocessor.mean.expand([batch_size, 3, 1, 1])
        stds = self.data_preprocessor.std.expand([batch_size, 3, 1, 1])

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().loss(
            src_data['inputs'], src_data['data_samples'], return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self.parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmengine.logging.print_log(f'Seg. Grad.: {grad_mag}')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(src_data['inputs'],
                                                      src_data['data_samples'],
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmengine.logging.print_log(f'Fdist Grad.: {grad_mag}')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            denormalized_target_image = torch.clamp(
                denorm(trg_data['inputs'], means, stds), 0, 1)

            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                trg_data['inputs'], trg_data['data_samples'])
            seg_debug['Target'] = self.get_ema_model().debug_output
            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Generate a relabeling map
            if self.local_iter == self.collect_mapping_info_start:
                self.collect_mapping_info = True
                print_log("START: Collect mapping info.", logger="current")
            elif self.local_iter == self.collect_mapping_info_end:
                for target_class, mapping_element in deepcopy(
                        self.relabeling_map.items()):
                    if mapping_element.meta_info[
                            'origin_of_source_classes'] == 'auto':
                        class_counts = mapping_element.source_candidates
                        print_log(
                            f"Print {target_class} class counts: {class_counts}",
                            logger="current")

                        class_to_category = self.patcher.clip_guide.get_class_index_to_category(
                        )

                        # Filter Counts - Remove not in same category
                        reduced_class_counts = Counter({
                            class_index: class_count
                            for class_index, class_count in
                            class_counts.items()
                            if class_to_category[class_index] ==
                            class_to_category[target_class]
                        })

                        # Get Source Class from Counts
                        if reduced_class_counts:
                            most_common_class = reduced_class_counts.most_common(
                                1)[0][0]

                            if most_common_class != target_class:
                                self.relabeling_map[
                                    target_class].source_class = most_common_class
                            else:
                                self.relabeling_map[
                                    target_class].source_class = None

                        # Remove element that does not have source classes
                        if self.relabeling_map[
                                target_class].source_class is None:
                            del self.relabeling_map[target_class]
                        else:
                            self.relabeling_map[target_class].meta_info[
                                'origin_of_source_classes'] = 'collected'

                print_log(
                    f"Generated relabeling map:\n{self.relabeling_map}",
                    logger="current")
                print_log(
                    "Relabeling map is generated by collected mapping info.",
                    logger="current")

                self.collect_mapping_info = False
                print_log("END: Collect mapping info.", logger="current")

                for index, _ in enumerate(self.bank_group):
                    if index not in self.relabeling_map.keys():
                        self.bank_group[index].clear()

            if not self.relabel_with_mapping_info and self.local_iter == self.relabel_with_mapping_info_start:
                if len(self.relabeling_map) == 0:
                    print_log(
                        'No element for relabeling. Do not relabel with mapping info.',
                        logger='current',
                        level=logging.WARNING)
                else:
                    self.relabel_with_mapping_info = True
                    print_log(
                        "START: Relabel with mapping info.", logger="current")
            elif self.relabel_with_mapping_info and self.local_iter == self.relabel_with_mapping_info_end:
                self.relabel_with_mapping_info = False
                print_log("END: Relabel with mapping info.", logger="current")

            # Collect or Relabel
            if self.collect_mapping_info or self.relabel_with_mapping_info:

                # Extract Patches - Detection
                batched_patches = self.patcher.extract_patches(
                    image=trg_data['inputs'],
                    label=pseudo_label.clone(),
                    weight=pseudo_weight,
                    mapping=self.relabeling_map)

                # Filter Patches - Validate by pseudo label
                if not self.collect_mapping_info:
                    batched_patches = self.patcher.filter_patches(
                        batched_patches=batched_patches,
                        validate_by='pseudo_label')

                # Filter Patches - Validate by patch size
                batched_patches = self.patcher.filter_patches(
                    batched_patches=batched_patches, validate_by='patch_size')

                # Classify Patches
                batched_patches = self.patcher.classify_patches(
                    batched_patches, self.relabeling_map)

                # Filter Patches - Validate by classification confidence
                batched_patches = self.patcher.filter_patches(
                    batched_patches=batched_patches,
                    validate_by='classification_confidence',
                    mapping=self.relabeling_map)

                src_image = self._stack_batch_gt(src_data['data_samples'])
                for batch, patches in enumerate(batched_patches):
                    for patch in patches:
                        patch.full_src_image = src_image[batch].clone()
                        # Collect Relations
                        if self.collect_mapping_info and self.relabeling_map[
                                patch.target_class].meta_info[
                                    'origin_of_source_classes'] == 'auto':
                            source_candidate = self.patcher.get_source_candidate(
                                patch=patch)
                            self.relabeling_map[
                                patch.
                                target_class].source_candidates += Counter(
                                    source_candidate)

                        # Paste Relabeled Patch - Relabel Patch with map
                        if self.relabel_with_mapping_info:
                            patch = self.patcher.relabel_patch(patch=patch)
                            pseudo_label[batch] = self.patcher.paste_patch(
                                patch=patch, pseudo_label=pseudo_label[batch])
                            self.total_relabled_count[patch.target_class] += 1
                            patch.full_relabeled_label = pseudo_label[
                                batch].clone()

                        # Save Patch to bank - Debugging
                        self.bank_group[patch.target_class].deposit_patch(
                            patch)

                # Debug Relabeled Patch Count
                if self.relabel_with_mapping_info:
                    log_vars.update({
                        f'relabeled_to.{target_class}': count
                        for target_class, count in enumerate(
                            self.total_relabled_count)
                    })

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            gt_semantic_seg = self._stack_batch_gt(
                src_data['data_samples'])  # To DO: Refact
            mix_masks = get_class_masks(gt_semantic_seg)

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack(
                        (src_data['inputs'][i], trg_data['inputs'][i])),
                    target=torch.stack(
                        (gt_semantic_seg[i][0], pseudo_label[i])))
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            del gt_pixel_weight

            mix_data = deepcopy(src_data)

            mix_data['inputs'] = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            for data_sample, lbl in zip(mix_data['data_samples'],
                                        mixed_lbl):  ## TO DO: REFACT
                data_sample.gt_sem_seg.data = lbl

            # Train on mixed images
            mix_losses = self.get_model().loss(
                mix_data['inputs'],
                mix_data['data_samples'],
                seg_weight=mixed_seg_weight,
                return_feat=False)
            seg_debug['Mix'] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self.parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()

            denormalized_mixed_image = torch.clamp(
                denorm(mix_data['inputs'], means, stds), 0, 1)

        # Masked training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), src_data, trg_data,
                                   valid_pseudo_mask, means, stds,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        # Debug results
        denormalized_source_image = torch.clamp(
            denorm(src_data['inputs'], means, stds), 0, 1)

        if self.debug_img:
            if seg_debug['Source'] is not None and seg_debug:
                if 'Target' in seg_debug:
                    seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
                    ).numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())

                    plots_info = [
                        dict(
                            axis=(k2, k1),
                            **prepare_debug_out(f'{n1} {n2}', out[j], means,
                                                stds))
                        for k1, (n1, outs) in enumerate(seg_debug.items())
                        for k2, (n2, out) in enumerate(outs.items())
                    ]

                    self.custom_debug[f'{j}_s'] = dict(
                        figure_info=dict(
                            shape=(rows, cols),
                            subplot_size=5,
                            kwargs=dict(
                                gridspec_kw={
                                    'hspace': 0.1,
                                    'wspace': 0,
                                    'top': 0.95,
                                    'bottom': 0,
                                    'right': 1,
                                    'left': 0
                                },
                                squeeze=False)),
                        plots_info=plots_info)
            del seg_debug

            for j in range(batch_size):
                if not self.source_only:
                    plots_info = [
                        dict(
                            axis=(0, 0),
                            image=denormalized_source_image[j],
                            title='Source Image'),
                        dict(
                            axis=(1, 0),
                            image=denormalized_target_image[j],
                            title='Target Image'),
                        dict(
                            axis=(0, 1),
                            image=gt_semantic_seg[j],
                            title='Source Seg GT',
                            cmap='cityscapes'),
                        dict(
                            axis=(1, 1),
                            image=pseudo_label[j],
                            title='Target Seg (Pseudo) GT',
                            cmap='cityscapes'),
                        dict(
                            axis=(0, 2),
                            image=denormalized_mixed_image[j],
                            title='Mixed Image'),
                        dict(
                            axis=(1, 2),
                            image=mix_masks[j][0],
                            title='Domain Mask',
                            cmap='gray'),
                        dict(
                            axis=(0, 3),
                            image=mixed_seg_weight[j],
                            title='Pseduo Weight')
                    ]

                    if mixed_lbl is not None:
                        plots_info.append(
                            dict(
                                axis=(1, 3),
                                image=mixed_lbl[j],
                                title='Seg Trg',
                                cmap='cityscapes'))

                    if self.debug_fdist_mask is not None:
                        plots_info.append(
                            dict(
                                axis=(0, 4),
                                image=self.debug_fdist_mask[j][0],
                                title='FDist Mask',
                                cmap='gray'))

                    if self.debug_gt_rescale is not None:
                        plots_info.append(
                            dict(
                                axis=(1, 4),
                                image=self.debug_gt_rescale[j],
                                title='Scaled GT',
                                cmap='cityscapes'))

                    self.custom_debug[f'{j}'] = dict(
                        figure_info=dict(shape=(2, 5)), plots_info=plots_info)

            # Patch Bank
            if self.enable_patching:
                if self.collect_mapping_info or self.relabel_with_mapping_info:
                    mean = self.data_preprocessor.mean.expand([3, 1, 1])
                    std = self.data_preprocessor.std.expand([3, 1, 1])
                    for target_class, patch_bank in enumerate(self.bank_group):
                        if len(patch_bank.vault) > 0:
                            plots_info = []
                            sum_of_height = 0
                            for i, patch in enumerate(patch_bank.vault):
                                sum_of_height += patch.image.shape[-2]
                                plots_info.extend([
                                    dict(
                                        axis=(i, 0),
                                        image=torch.clamp(
                                            denorm(patch.image, mean, std), 0,
                                            1),
                                        title='Target Cropped Image'
                                        if i == 0 else None),
                                    dict(
                                        axis=(i, 1),
                                        image=patch.pseudo_label,
                                        title=
                                        f'from: {patch.source_class} \n (det_conf: {patch.detection_confidence:.4f}) \n (cls_conf: {patch.classification_confidence})',
                                        cmap='cityscapes')
                                ])

                                if patch.segment_mask is not None:
                                    plots_info.append(
                                        dict(
                                            axis=(i, 2),
                                            image=patch.segment_mask,
                                            title='Target Segment Mask'
                                            if i == 0 else None,
                                            cmap='gray'))

                                if patch.relabeled_label is not None:
                                    plots_info.append(
                                        dict(
                                            axis=(i, 3),
                                            image=patch.relabeled_label,
                                            title='Target Relabeled Pseudo Label'
                                            if i == 0 else None,
                                            cmap='cityscapes'))

                                if patch.full_pseudo_label is not None:
                                    plots_info.append(
                                        dict(
                                            axis=(i, 4),
                                            image=patch.full_pseudo_label,
                                            title='Target Full Pseudo Label'
                                            if i == 0 else None,
                                            cmap='cityscapes'))

                                if patch.full_relabeled_label is not None:
                                    plots_info.append(
                                        dict(
                                            axis=(i, 5),
                                            image=patch.full_relabeled_label,
                                            title='Target Full Relabeled Label'
                                            if i == 0 else None,
                                            cmap='cityscapes'))
                                if patch.full_src_image is not None:
                                    plots_info.append(
                                        dict(
                                            axis=(i, 6),
                                            image=torch.clamp(
                                                denorm(patch.full_src_image,
                                                       mean, std), 0, 1),
                                            title='Source Full Image'
                                            if i == 0 else None))
                                if patch.full_image is not None:
                                    plots_info.append(
                                        dict(
                                            axis=(i, 6),
                                            image=torch.clamp(
                                                denorm(patch.full_image, mean,
                                                       std), 0, 1),
                                            title='Target Full Image'
                                            if i == 0 else None))

                                if i == 0:
                                    sum_of_width = patch.image.shape[-1] * 4
                                    number_of_columns = 8

                            figure_width = 3 * number_of_columns
                            figure_height = figure_width * sum_of_height / sum_of_width
                            self.custom_debug[
                                f'Patch_Bank[{target_class}] (All)'] = dict(
                                    figure_info=dict(
                                        shape=(len(patch_bank.vault),
                                               number_of_columns)),
                                    plots_info=plots_info)

        return log_vars
