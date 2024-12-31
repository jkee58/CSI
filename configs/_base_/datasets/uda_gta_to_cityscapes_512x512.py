# Obtained from: https://github.com/lhoyer/MIC
# Modifications: Migration from MMSegmentation 0.x to 1.x
# ---------------------------------------------------------------
# Copyright (c) 2024 Sungkyunkwan University, Jeongkee Lim. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Dataset settings
crop_size = (512, 512)
gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1280, 720), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='PackSegInputs')
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
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
            type='GTADataset',
            data_root='data/gta/',
            data_prefix=dict(img_path='images', seg_map_path='labels'),
            pipeline=gta_train_pipeline),
        target=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            data_prefix=dict(
                img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
            pipeline=cityscapes_train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
