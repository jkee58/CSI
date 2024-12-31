# Dataset settings
crop_size = (512, 512)
synthia_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1280, 760), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='PackSegInputs')
]

acdc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 540),
         keep_ratio=True),  # original 1920x1080
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 540), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(960, 540), keep_ratio=True),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='UDADataset',
        source=dict(
            type='SynthiaDataset',
            data_root='data/synthia/',
            data_prefix=dict(img_path='RGB', seg_map_path='GT/LABELS'),
            pipeline=synthia_train_pipeline),
        target=dict(
            type='ACDCDataset',
            data_root='data/acdc/',
            data_prefix=dict(
                img_path='rgb_anon/train', seg_map_path='gt/train'),
            pipeline=acdc_train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ACDCDataset',
        data_root='data/acdc/',
        data_prefix=dict(img_path='rgb_anon/val', seg_map_path='gt/val'),
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ACDCDataset',
        data_root='data/acdc/',
        data_prefix=dict(img_path='rgb_anon/test', seg_map_path='gt/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], format_only=True)
