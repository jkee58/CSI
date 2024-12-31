# Obtained from: https://github.com/lhoyer/MIC
# Modifications: Migration from MMSegmentation 0.x to 1.x
# ---------------------------------------------------------------
# Copyright (c) 2024 Sungkyunkwan University, Jeongkee Lim. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# Learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=40000,
        eta_min=0.0,
        by_epoch=False)
]
