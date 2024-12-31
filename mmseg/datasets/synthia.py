# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.registry import DATASETS
from . import CityscapesDataset
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class SynthiaDataset(BaseSegDataset):
    METAINFO = dict(
        classes = CityscapesDataset.METAINFO['classes'],
        palette = CityscapesDataset.METAINFO['palette'])

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(SynthiaDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            **kwargs)
