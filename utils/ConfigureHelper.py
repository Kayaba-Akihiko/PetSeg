#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch
import numpy as np
import random
import os

# try to compute a suggested max number of worker based on system's resource
_max_num_worker_suggest = 0
if hasattr(os, 'sched_getaffinity'):
    try:
        _max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
if _max_num_worker_suggest == 0:
    # os.cpu_count() could return Optional[int]
    # get cpu count first and check None in order to satify mypy check
    cpu_count = os.cpu_count()
    if cpu_count is not None:
        _max_num_worker_suggest = cpu_count


class ConfigureHelper:
    max_n_workers = _max_num_worker_suggest
    float_point = "32"

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    TQDM_INTERVAL = (10, 60)

