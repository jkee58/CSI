# Obtained from: https://github.com/lhoyer/MIC
# Modifications: Migration from MMSegmentation 0.x to 1.x
# ---------------------------------------------------------------
# Copyright (c) 2024 Sungkyunkwan University, Jeongkee Lim. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

optim_wrapper = dict(
    optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))
