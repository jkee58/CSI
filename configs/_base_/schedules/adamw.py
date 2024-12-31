# Obtained from: https://github.com/lhoyer/MIC
# Modifications: Migration from MMSegmentation 0.x to 1.x
# ---------------------------------------------------------------
# Copyright (c) 2024 Sungkyunkwan University, Jeongkee Lim. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=0.00006,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.01))
