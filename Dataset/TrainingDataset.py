#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from .BaseDataset import BaseDataset
from typing import Union
from utils.ConfigureHelper import ConfigureHelper
import numpy as np


class TrainingDataset(BaseDataset):
    def __init__(self,
                 data_root: str,
                 preload_dataset: bool,
                 ret_dsize: Union[int, tuple[int, int]],
                 n_preload_worker=ConfigureHelper.max_n_workers):
        super(TrainingDataset, self).__init__(data_root=data_root,
                                              split="trainval",
                                              preload_dataset=preload_dataset,
                                              ret_dsize=ret_dsize,
                                              n_preload_worker=n_preload_worker)

    def _augment(self, image, seg):
        if np.random.random() < 0.5:
            image = np.flip(image, axis=0)
            seg = np.flip(seg, axis=0)
        if np.random.random() < 0.5:
            image = np.flip(image, axis=1)
            seg = np.flip(seg, axis=1)
        return image, seg