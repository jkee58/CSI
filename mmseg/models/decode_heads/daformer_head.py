# Obtained from https://github.com/lhoyer/MIC

from typing import Tuple
from overrides import overrides

import torch
from torch import Tensor
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.utils import OptSampleList, ConfigType, SampleList
from ..utils import resize
from ..builder import HEADS
from .aspp_head import ASPPModule
from .decode_head import BaseDecodeHead
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


@HEADS.register_module()
class DAFormerHead(BaseDecodeHead):

    def __init__(self, extract_visual_feat=False, **kwargs):
        super(DAFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

        self.extract_visual_feat = extract_visual_feat

    def forward(self, inputs):
        if self.extract_visual_feat:
            inputs = inputs[0][0]
        x = inputs

        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        return x

    def forward_module(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        return x

    def convert_gt(self, seg_logits, batch_data_samples):
        i_seg_pred = self.convert_seg_logits_to_pred(seg_logits)
        return batch_data_samples

    def convert_seg_logits_to_pred(
            self,
            seg_logits: Tensor,
            data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to pred.
        """
        batch_size, C, H, W = seg_logits.shape
        seg_preds = []
        for i in range(batch_size):
            i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
                seg_preds.append(i_seg_pred)
            else:
                raise NotImplementedError(C)

        # seg_logits = resize(
        #     input=seg_logits,
        #     size=batch_img_metas[0]['img_shape'],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return seg_preds

    @overrides
    def loss(self,
             inputs: Tuple[Tensor],
             batch_data_samples: SampleList,
             train_cfg: ConfigType,
             seg_weight=None,
             return_logits=False) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.debug_output = {}
        seg_logits = self.forward(inputs)
        # convert_16_to_19 = True
        # if convert_16_to_19:
        #     if seg_logits == truck_index and gt_sem_seg == car_index
        #         gt_sem_seg = truck_index
        #     gt_sem_seg == car_index
        #     pass

        losses = self.loss_by_feat(seg_logits, batch_data_samples, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        return losses

    # @overrides
    # def loss(self,
    #          inputs,
    #          batch_data_samples,
    #          train_cfg,
    #          seg_weight=None,
    #          return_logits=False,
    #          return_code=False,
    #          loss_name='loss_cross_entropy',
    #          **kwargs) -> dict:
    #     if loss_name == 'loss_stego':
    #         loss_target = kwargs['loss_target']
    #         H, W = 128, 128
    #         # feats = torch.cat([resize(feat, size=(H, W)) for feat in inputs],
    #         #                   axis=1)
    #         feats = resize(inputs[-1], size=(H, W))
    #         code = self.forward(inputs, return_code=True)

    #         if loss_target == 'positive_inter':
    #             losses = self.compute_contrastive_loss(orig_feats_pos=feats,
    #                                                    orig_code_pos=code,
    #                                                    **kwargs)
    #         else:
    #             losses = {
    #                 loss_target:
    #                 self.loss_stego(orig_feats=feats, orig_code=code, **kwargs)
    #             }
    #         losses = add_prefix(losses, loss_name)
    #     elif loss_name == 'loss_linear':
    #         with torch.no_grad():
    #             code = self.forward(inputs, return_code=True)
    #         detached_code = torch.clone(code.detach())

    #         linear_logits = self.linear_probe(detached_code)
    #         seg_label = self._stack_batch_gt(batch_data_samples)
    #         linear_logits = resize(input=linear_logits,
    #                                size=seg_label.shape[2:],
    #                                mode='bilinear',
    #                                align_corners=self.align_corners)
    #         seg_label = seg_label.squeeze(1)

    #         losses = {
    #             loss_name:
    #             self.loss_cross_entropy(linear_logits,
    #                                     seg_label,
    #                                     weight=seg_weight,
    #                                     ignore_index=self.ignore_index)
    #         }
    #     elif loss_name == 'loss_cross_entropy':
    #         seg_logits = self.forward(inputs, return_code=False)
    #         seg_label = self._stack_batch_gt(batch_data_samples)
    #         seg_logits = resize(input=seg_logits,
    #                             size=seg_label.shape[2:],
    #                             mode='bilinear',
    #                             align_corners=self.align_corners)
    #         seg_label = seg_label.squeeze(1)

    #         losses = {
    #             loss_name:
    #             self.loss_cross_entropy(seg_logits,
    #                                     seg_label,
    #                                     weight=seg_weight,
    #                                     ignore_index=self.ignore_index)
    #         }
    #     else:
    #         seg_logits = self.forward(inputs, return_code)
    #         losses = self.loss_by_feat(seg_logits, batch_data_samples,
    #                                    seg_weight)
    #     return losses

    # @overrides
    # def loss_by_feat(self,
    #                  seg_logits: Tensor,
    #                  batch_data_samples: SampleList,
    #                  seg_weight=None) -> dict:
    #     loss = dict()
    #     seg_label = self._stack_batch_gt(batch_data_samples)
    #     seg_logits = resize(input=seg_logits,
    #                         size=seg_label.shape[2:],
    #                         mode='bilinear',
    #                         align_corners=self.align_corners)

    #     if self.sampler is not None:
    #         seg_weight = self.sampler.sample(seg_logits, seg_label)

    #     seg_label = seg_label.squeeze(1)

    #     if not isinstance(self.loss_decode, nn.ModuleList):
    #         losses_decode = [self.loss_decode]
    #     else:
    #         losses_decode = self.loss_decode

    #     for loss_decode in losses_decode:
    #         loss_decode.debug = self.debug
    #         if loss_decode.loss_name not in loss:
    #             loss[loss_decode.loss_name] = loss_decode(
    #                 seg_logits,
    #                 seg_label,
    #                 weight=seg_weight,
    #                 ignore_index=self.ignore_index)
    #         else:
    #             loss[loss_decode.loss_name] += loss_decode(
    #                 seg_logits,
    #                 seg_label,
    #                 weight=seg_weight,
    #                 ignore_index=self.ignore_index)

    #     loss['acc_seg'] = accuracy(seg_logits,
    #                                seg_label,
    #                                ignore_index=self.ignore_index)

    #     for loss_decode in losses_decode:
    #         if loss_decode.loss_name != 'loss_stego':
    #             if self.debug and hasattr(loss_decode, 'debug_output'):
    #                 self.debug_output.update(loss_decode.debug_output)

    #     return loss
