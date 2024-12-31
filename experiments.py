import itertools
import os
import time

from mmengine.config import Config
from nvitop import Device

# flake8: noqa


def get_model_base(architecture: str, backbone: str):
    base_model_path = '_base_/models'
    architecture = architecture.replace('sfa_', '')
    for j in range(1, 100):
        hrda_name = [e for e in architecture.split('_') if f'hrda{j}' in e]
        for n in hrda_name:
            architecture = architecture.replace(f'{n}_', '')

    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'

    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}', style='pytorch')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        }
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature, min_crop_ratio):
    cfg.setdefault('train_dataloader', {}).setdefault('dataset', {})
    cfg['train_dataloader']['dataset']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=min_crop_ratio)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        base_schedule_path = '_base_/schedules'
        cfg = {
            '_base_': ['_base_/default_runtime.py'],
            'gpu_model': gpu_model,
            'n_gpus': n_gpus,
            'gpu_index': gpu_index
        }
        if seed is not None:
            cfg['randomness'] = dict(seed=seed)

        # Setup model config
        architecture_mod = architecture
        sync_crop_size_mod = sync_crop_size
        inference_mod = inference
        model_base = get_model_base(architecture_mod, backbone)
        model_base_cfg = Config.fromfile(os.path.join('configs', model_base))
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }

        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        hrda_ablation_opts = None
        outer_crop_size = sync_crop_size_mod \
            if sync_crop_size_mod is not None \
            else (int(crop.split('x')[0]), int(crop.split('x')[1]))

        if 'hrda1' in architecture_mod:
            o = [e for e in architecture_mod.split('_') if 'hrda' in e][0]
            hr_crop_size = (int((o.split('-')[1])), int((o.split('-')[1])))
            hr_loss_w = float(o.split('-')[2])
            hrda_ablation_opts = o.split('-')[3:]
            cfg['model']['type'] = 'HRDAEncoderDecoder'
            cfg['model']['scales'] = [1, 0.5]
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['single_scale_head'] = model_base_cfg[
                'model']['decode_head']['type']
            cfg['model']['decode_head']['type'] = 'HRDAHead'
            cfg['model']['hr_crop_size'] = hr_crop_size
            cfg['model']['feature_scale'] = 0.5
            cfg['model']['crop_coord_divisible'] = 8
            cfg['model']['hr_slide_inference'] = True
            cfg['model']['decode_head']['attention_classwise'] = True
            cfg['model']['decode_head']['hr_loss_weight'] = hr_loss_w
            if outer_crop_size == hr_crop_size:
                # If the hr crop is smaller than the lr crop (hr_crop_size <
                # outer_crop_size), there is direct supervision for the lr
                # prediction as it is not fused in the region without hr
                # prediction. Therefore, there is no need for a separate
                # lr_loss.
                cfg['model']['decode_head']['lr_loss_weight'] = hr_loss_w
                # If the hr crop covers the full lr crop region, calculating
                # the FD loss on both scales stabilizes the training for
                # difficult classes.
                cfg['model']['feature_scale'] = 'all' if '_fd' in uda else 0.5

        # HRDA Ablations
        if hrda_ablation_opts is not None:
            for o in hrda_ablation_opts:
                if o == 'fixedatt':
                    # Average the predictions from both scales instead of
                    # learning a scale attention.
                    cfg['model']['decode_head']['fixed_attention'] = 0.5
                elif o == 'nooverlap':
                    # Don't use overlapping slide inference for the hr
                    # prediction.
                    cfg['model']['hr_slide_overlapping'] = False
                elif o == 'singleatt':
                    # Use the same scale attention for all class channels.
                    cfg['model']['decode_head']['attention_classwise'] = False
                elif o == 'blurhr':
                    # Use an upsampled lr crop (blurred) for the hr crop
                    cfg['model']['blur_hr_crop'] = True
                elif o == 'samescale':
                    # Use the same scale/resolution for both crops.
                    cfg['model']['scales'] = [1, 1]
                    cfg['model']['feature_scale'] = 1
                elif o[:2] == 'sc':
                    cfg['model']['scales'] = [1, float(o[2:])]
                    if not isinstance(cfg['model']['feature_scale'], str):
                        cfg['model']['feature_scale'] = float(o[2:])
                else:
                    raise NotImplementedError(o)

        # Setup inference mode
        if inference_mod == 'whole' or crop == '2048x1024':
            assert model_base_cfg['model']['test_cfg']['mode'] == 'whole'
        elif inference_mod == 'slide':
            cfg['model'].setdefault('test_cfg', {})
            cfg['model']['test_cfg']['mode'] = 'slide'
            cfg['model']['test_cfg']['batched_slide'] = True
            crsize = sync_crop_size_mod if sync_crop_size_mod is not None \
                else [int(e) for e in crop.split('x')]
            cfg['model']['test_cfg']['stride'] = [e // 2 for e in crsize]
            cfg['model']['test_cfg']['crop_size'] = crsize
            architecture_mod += '_sl'
        else:
            raise NotImplementedError(inference_mod)

        # Setup data preprocessor
        cfg['model']['data_preprocessor'] = dict(
            type='SegDataPreProcessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            size=(512, 512),
            pad_val=0,
            seg_pad_val=255)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')

        # DAFormer legacy cropping that only works properly if the training
        # crop has the height of the (resized) target image.
        if ('dacs' in uda or mask_mode is not None) and plcrop in [True, 'v1']:
            cfg.setdefault('uda', {})
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        if 'dacs' in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T, rcs_min_crop)
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        if ('dacs' in uda or mask_mode is not None) and plcrop == 'v2':
            cfg['train_dataloader']['dataset'].setdefault('target', {})
            cfg['train_dataloader']['dataset']['target']['crop_pseudo_margins'] = \
                [30, 240, 30, 30]
        if 'dacs' in uda and sync_crop_size_mod is not None:
            cfg.setdefault('train_dataloader', {}).setdefault('dataset', {})
            cfg['train_dataloader']['dataset'][
                'sync_crop_size'] = sync_crop_size_mod
            cfg['train_dataloader']['dataset'][
                'sync_crop_size'] = sync_crop_size_mod
        if mask_mode is not None:
            cfg.setdefault('uda', {})
            cfg['uda']['mask_mode'] = mask_mode
            cfg['uda']['mask_alpha'] = mask_alpha
            cfg['uda']['mask_pseudo_threshold'] = mask_pseudo_threshold
            cfg['uda']['mask_lambda'] = mask_lambda
            cfg['uda']['mask_generator'] = dict(
                type='block',
                mask_ratio=mask_ratio,
                mask_block_size=mask_block_size,
                _delete_=True)
        else:
            cfg.setdefault('uda', {})

        # Setup CLIP-guided
        cfg['uda'][
            'range_of_relabel_with_mapping_info'] = range_of_relabel_with_mapping_info
        cfg['uda'][
            'range_of_collect_mapping_info'] = range_of_collect_mapping_info
        cfg['uda']['patch_bank_capacity'] = patch_bank_capacity
        cfg['uda']['ckpt_for_obj_det'] = ckpt_for_obj_det
        cfg['uda']['relabeling_cfg'] = relabeling_cfg

        relabel_info = 'relabel_' + str(
            cfg['uda']['range_of_relabel_with_mapping_info'][0]
        ) + '_to_' + str(cfg['uda']['range_of_relabel_with_mapping_info'][1])

        # Setup optimizer and schedule
        if 'dacs' in uda or 'minent' in uda or 'advseg' in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].append(f'_base_/schedules/{opt}.py')
        if schedule is not None:
            cfg['_base_'].append(f'_base_/schedules/{schedule}.py')
        cfg.setdefault('optim_wrapper', {})
        cfg['optim_wrapper']['optimizer'] = {'lr': lr}
        cfg['optim_wrapper'].setdefault('paramwise_cfg', {})
        cfg['optim_wrapper']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optim_wrapper']['paramwise_cfg'][
            'custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner_type'] = runner_type
        cfg['train_cfg'] = dict(
            type='IterBasedTrainLoop',
            max_iters=iters,
            val_interval=iters // 10)
        cfg['val_cfg'] = dict(type='ValLoop')
        cfg['test_cfg'] = dict(type='TestLoop')

        cfg['val_evaluator'] = dict(type='IoUMetric', iou_metrics=['mIoU'])
        cfg['test_evaluator'] = cfg['val_evaluator']

        cfg.setdefault('train_dataloader', {})
        cfg['train_dataloader']['batch_size'] = batch_size

        # Setup default_hooks
        cfg['default_hooks'] = dict(
            logger=dict(
                type='LoggerHook', interval=50, log_metric_by_epoch=False),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(
                type='CheckpointHook',
                by_epoch=False,
                interval=iters,
                save_best='mIoU'),
            visualization=dict(
                type='SegVisualizationHook', draw=True, interval=50))

        # Setup custom hooks
        if use_custom_hook:
            cfg['custom_hooks'] = [
                dict(
                    type='CustomDebugHook',
                    priority='ABOVE_NORMAL',
                    interval=1000),
                dict(
                    type='WandbCommitHook',
                    priority='BELOW_NORMAL',
                    interval=1)
            ]

        # Setup visualizer name
        # Name must be different in other seed for running in one program
        cfg['visualizer'] = dict(name=f'visualizer_{str(time.time())}')

        # Construct config name
        uda_mod = uda
        if 'dacs' in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
            if rcs_min_crop != 0.5:
                uda_mod += f'-{rcs_min_crop}'
        if 'dacs' in uda and sync_crop_size_mod is not None:
            uda_mod += f'_sf{sync_crop_size_mod[0]}x{sync_crop_size_mod[1]}'
        if 'dacs' in uda or mask_mode is not None:
            if not plcrop:
                pass
            elif plcrop in [True, 'v1']:
                uda_mod += '_cpl'
            elif plcrop[0] == 'v':
                uda_mod += f'_cpl{plcrop[1:]}'
            else:
                raise NotImplementedError(plcrop)
        if mask_mode is not None:
            uda_mod += f'_m{mask_block_size}-' \
                       f'{mask_ratio}-'
            if mask_alpha != 'same':
                uda_mod += f'a{mask_alpha}-'
            if mask_pseudo_threshold != 'same':
                uda_mod += f'p{mask_pseudo_threshold}-'
            uda_mod += {
                'separate': 'sep',
                'separateaug': 'spa',
                'separatesrc': 'sps',
                'separatesrcaug': 'spsa',
                'separatetrg': 'spt',
                'separatetrgaug': 'spta',
            }[mask_mode]
            if mask_lambda != 1:
                uda_mod += f'-w{mask_lambda}'
        crop_name = f'_{crop}' if crop != '512x512' else ''
        cfg['name'] = f'{source}2{target}{crop_name}_{uda_mod}_' \
                      f'{architecture_mod}_{backbone}_{schedule}_' \
                      f'{relabel_info}'
        if opt != 'adamw':
            cfg['name'] += f'_{opt}'
        if lr != 0.00006:
            cfg['name'] += f'_{lr}'
        if not pmult:
            cfg['name'] += f'_pm{pmult}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}{crop_name}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('None', 'N').replace('[', '')\
            .replace(']', '').replace(',', 'j').replace(' ', '') \
            .replace('cityscapes', 'cs') \
            .replace('synthia', 'syn') \
            .replace('darkzurich', 'dzur') \
            .replace('relabel', 'rl')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    n_gpus = 1
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'

    # Select the first single gpu as default
    devices = Device.cuda.all()
    gpu_model = devices[0].name()
    gpu_index = devices[0].nvml_index

    architecture = None
    rcs_T = None
    rcs_min_crop = 0.5
    plcrop = False
    inference = 'whole'
    sync_crop_size = None
    mask_mode = None
    mask_alpha = 'same'
    mask_pseudo_threshold = 'same'
    mask_lambda = 1
    mask_block_size = None
    mask_ratio = 0

    runner_type = 'CustomRunner'
    use_custom_hook = True

    # Default configuration for CSI
    range_of_relabel_with_mapping_info = (-1, -1)
    range_of_collect_mapping_info = (-1, -1)
    patch_bank_capacity = 25
    ckpt_for_obj_det = 'owlvit'
    relabeling_cfg = None

    # -------------------------------------------------------------------------
    # MIC with Further UDA Methods
    # -------------------------------------------------------------------------
    # yapf: disable
    if id == 81:
        seeds = [0, 1, 2]
        #        opt,     lr,      schedule,     pmult
        sgd   = ('sgd',   0.0025,  'poly10warm', False)
        adamw = ('adamw', 0.00006, 'poly10warm', True)
        #               uda,                  rcs_T, plcrop, opt_hp
        uda_advseg =   ('advseg',             None,  False,  *sgd)
        uda_minent =   ('minent',             None,  False,  *sgd)
        uda_dacs =     ('dacs',               None,  False,  *adamw)
        uda_daformer = ('dacs_a999_fdthings', 0.01,  True,   *adamw)
        uda_hrda =     ('dacs_a999_fdthings', 0.01,  'v2',   *adamw)
        mask_mode, mask_ratio = 'separatetrgaug', 0.7
        for architecture,                      backbone,  uda_hp in [
            # ('dlv2red',                        'r101v1c', uda_advseg),
            # ('dlv2red',                        'r101v1c', uda_minent),
            # ('dlv2red',                        'r101v1c', uda_dacs),
            ('dlv2red',                        'r101v1c', uda_daformer),
            # ('hrda1-512-0.1_dlv2red',          'r101v1c', uda_hrda),
            # ('daformer_sepaspp',               'mitb5',   uda_daformer),
            # ('hrda1-512-0.1_daformer_sepaspp', 'mibt5',   uda_hrda),  # already run in exp 80
        ]:
            if 'hrda' in architecture:
                source, target, crop = 'gtaHR', 'cityscapesHR', '1024x1024'
                rcs_min_crop = 0.5 * (2 ** 2)
                inference = 'slide'
                mask_block_size = 64
            else:
                source, target, crop = 'gta', 'cityscapes', '512x512'
                rcs_min_crop = 0.5
                inference = 'whole'
                # Use half the patch size when training with half resolution
                mask_block_size = 32
            for seed in seeds:
                uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Comparsion with SoTA Study (Table 1)
    # -------------------------------------------------------------------------
    elif id == 90:
        seeds = [0, 1, 2]
        ckpt_for_obj_det = 'owlvit'

        new_class = [
            # nature
            (9, 8, ('detect', 0.01, 0.1)),
            # vehicle
            (14, 13, ('detect', 0.1, 0.5)),
            (16, None, ('detect', 0.1, 0.5)),
        ]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        adamw = ('adamw', 0.00006, 'poly10warm', True)
        uda_daformer = ('dacs_a999_fdthings', 0.01, True, *adamw)
        uda_hrda = ('dacs_a999_fdthings', 0.01, 'v2', *adamw)
        uda_mic = ('separate', 0.7)
        relabeling_cfg = new_class
        for architecture, backbone, uda_hp, mask_hp, range_of_collect_mapping_info, range_of_relabel_with_mapping_info in [
            ('daformer_sepaspp', 'mitb5', uda_daformer, None, (-1, -1), (-1, -1)),
            ('daformer_sepaspp', 'mitb5', uda_daformer, None, (7999, 11999), (11999, 40000)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, None, (-1, -1), (-1, -1)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, None, (7999, 11999), (11999, 40000)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, uda_mic, (-1, -1), (-1, -1)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, uda_mic, (7999, 11999), (11999, 39999))
        ]:

            if 'hrda' in architecture:
                source, target, crop = 'synthiaHR', 'cityscapesHR', '1024x1024'
                rcs_min_crop = 0.5 * (2**2)
                inference = 'slide'
            else:
                source, target, crop = 'synthia', 'cityscapes', '512x512'
                rcs_min_crop = 0.5
                inference = 'whole'

            if mask_hp is not None:
                mask_mode, mask_ratio = mask_hp
                if 'hrda' in architecture:
                    mask_block_size = 64
                else:
                    mask_block_size = 32
            for seed in seeds:
                uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Ablation Study (Table 2)
    # -------------------------------------------------------------------------
    elif id == 91:
        seeds = [0, 1, 2]

        new_classes = [
            # nature
            (9, 8, ('detect', 0.01, 0.1)),
            # vehicle
            (14, 13, ('detect', 0.1, 0.5)),
            (16, None, ('detect', 0.1, 0.5)),
        ]

        adamw = ('adamw', 0.00006, 'poly10warm', True)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for source, target, crop, architecture, backbone, uda, opt_hp, rlbl_range in [
            ('synthia', 'cityscapes', '512x512', 'dlv2red', 'r101v1c', 'dacs', adamw, (-1, -1)),
            ('synthia', 'cityscapes', '512x512', 'dlv2red', 'r101v1c', 'dacs', adamw, (11999, 39999)),
            ('synthia', 'cityscapes', '512x512', 'dlv2red', 'r101v1c', 'daformer_uda', adamw, (-1, -1)),
            ('synthia', 'cityscapes', '512x512', 'dlv2red', 'r101v1c', 'daformer_uda', adamw, (11999, 39999)),
            ('synthiaHR', 'cityscapesHR', '1024x1024', 'hrda1-512-0.1_daformer_sepaspp', 'mitb5', 'hrda_uda',  adamw, (11999, 39999))
        ]:
            if uda == 'daformer_uda':
                uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, True
                if crop == '1024x1024':
                    rcs_min_crop = 0.5 * (2 ** 2)
            if uda == 'hrda_uda':
                uda, rcs_T, plcrop = 'dacs_a999_fdthings', 0.01, 'v2'
                rcs_min_crop = 0.5 * (2**2)
                inference = 'slide'
            range_of_relabel_with_mapping_info = rlbl_range
            relabeling_cfg = new_classes
            for seed in seeds:
                opt, lr, schedule, pmult = opt_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Ablation Study (Table 3) (No CLIP Filtering)
    # -------------------------------------------------------------------------
    elif id == 92:
        seeds = [0, 1, 2]
        ckpt_for_obj_det = 'owlvit'

        new_classes = [
            # nature
            (9, 8, ('detect', 0.01, None)),
            # vehicle
            (14, 13, ('detect', 0.1, None)),
            (16, None, ('detect', 0.1, None)),
        ]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        adamw = ('adamw', 0.00006, 'poly10warm', True)
        uda_daformer = ('dacs_a999_fdthings', 0.01, True, *adamw)
        uda_hrda = ('dacs_a999_fdthings', 0.01, 'v2', *adamw)
        uda_mic = ('separate', 0.7)
        relabeling_cfg = new_classes
        for architecture, backbone, uda_hp, mask_hp, range_of_collect_mapping_info, range_of_relabel_with_mapping_info in [
            ('daformer_sepaspp', 'mitb5', uda_daformer, None, (7999, 11999), (11999, 40000))
        ]:

            if 'hrda' in architecture:
                source, target, crop = 'synthiaHR', 'cityscapesHR', '1024x1024'
                rcs_min_crop = 0.5 * (2**2)
                inference = 'slide'
            else:
                source, target, crop = 'synthia', 'cityscapes', '512x512'
                rcs_min_crop = 0.5
                inference = 'whole'

            if mask_hp is not None:
                mask_mode, mask_ratio = mask_hp
                if 'hrda' in architecture:
                    mask_block_size = 64
                else:
                    mask_block_size = 32
            for seed in seeds:
                uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Ablation Study (Table 4) (GTA-to-Cityscapes)
    # -------------------------------------------------------------------------
    elif id == 93:
        seeds = [0, 1, 2]
        ckpt_for_obj_det = 'owlvit'

        new_classes = [
            # nature
            (9, 8, ('detect', 0.01, 0.1)),
            # vehicle
            (14, 13, ('detect', 0.1, 0.5)),
            (16, None, ('detect', 0.1, 0.5)),
        ]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        adamw = ('adamw', 0.00006, 'poly10warm', True)
        uda_daformer = ('dacs_a999_fdthings', 0.01, True, *adamw)
        uda_hrda = ('dacs_a999_fdthings', 0.01, 'v2', *adamw)
        uda_mic = ('separate', 0.7)
        relabeling_cfg = new_classes
        for architecture, backbone, uda_hp, mask_hp, range_of_collect_mapping_info, range_of_relabel_with_mapping_info in [
            ('daformer_sepaspp', 'mitb5', uda_daformer, None, (-1, -1), (-1, -1)),
            ('daformer_sepaspp', 'mitb5', uda_daformer, None, (7999, 11999), (11999, 40000)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, uda_mic, (-1, -1), (-1, -1)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, uda_mic, (7999, 11999), (11999, 39999))
        ]:

            if 'hrda' in architecture:
                source, target, crop = 'gta16HR', 'cityscapesHR', '1024x1024'
                rcs_min_crop = 0.5 * (2**2)
                inference = 'slide'
            else:
                source, target, crop = 'gta16', 'cityscapes', '512x512'
                rcs_min_crop = 0.5
                inference = 'whole'
            if mask_hp is not None:
                mask_mode, mask_ratio = mask_hp
                if 'hrda' in architecture:
                    mask_block_size = 64
                else:
                    mask_block_size = 32
            for seed in seeds:
                uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Ablation Study (Table 5)
    # -------------------------------------------------------------------------
    elif id == 94:
        seeds = [0, 1, 2]
        ckpt_for_obj_det = 'owlvit'

        new_classes = [
            # nature
            (9, 8, ('detect', 0.01, 0.1)),
            # vehicle
            (14, 13, ('detect', 0.1, 0.5)),
            (16, None, ('detect', 0.1, 0.5)),
        ]

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        adamw = ('adamw', 0.00006, 'poly10warm', True)
        uda_daformer = ('dacs_a999_fdthings', 0.01, True, *adamw)
        uda_hrda = ('dacs_a999_fdthings', 0.01, 'v2', *adamw)
        uda_mic = ('separate', 0.7)
        relabeling_cfg = new_classes
        for architecture, backbone, uda_hp, mask_hp, range_of_collect_mapping_info, range_of_relabel_with_mapping_info in [
            ('daformer_sepaspp', 'mitb5', uda_daformer, None, (-1, -1), (-1, -1)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, None, (-1, -1), (-1, -1)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, uda_mic, (-1, -1), (-1, -1)),
            ('hrda1-512-0.1_daformer_sepaspp', 'mitb5', uda_hrda, uda_mic, (7999, 11999), (11999, 39999))
        ]:
            if 'hrda' in architecture:
                source, target, crop = 'synthiaHR', 'acdcHR', '1024x1024'
                rcs_min_crop = 0.5 * (2**2)
                inference = 'slide'
            else:
                source, target, crop = 'synthia', 'acdc', '512x512'
                rcs_min_crop = 0.5
                inference = 'whole'
            if mask_hp is not None:
                mask_mode, mask_ratio = mask_hp
                if 'hrda' in architecture:
                    mask_block_size = 64
                else:
                    mask_block_size = 32
            for seed in seeds:
                uda, rcs_T, plcrop, opt, lr, schedule, pmult = uda_hp
                cfg = config_from_vars()
                cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))
    return cfgs
