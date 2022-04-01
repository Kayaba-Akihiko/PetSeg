#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from .BaseDataset import BaseDataset
from typing import Union
from random import randint


class VisualizationDataset(BaseDataset):
    def __init__(self,
                 data_root: str,
                 ret_dsize: Union[int, tuple[int, int]],
                 ):
        super(VisualizationDataset, self).__init__(data_root=data_root,
                                                   split="test",
                                                   preload_dataset=False,
                                                   ret_dsize=ret_dsize,)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        idx = randint(0, super(VisualizationDataset, self).__len__())
        return super(VisualizationDataset, self).__getitem__(idx=idx)
