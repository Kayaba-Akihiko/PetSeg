#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from .BaseDataset import BaseDataset
from typing import Union
from utils.ConfigureHelper import ConfigureHelper


class TestDataset(BaseDataset):
    def __init__(self,
                 data_root: str,
                 preload_dataset: bool,
                 ret_dsize: Union[int, tuple[int, int]],
                 n_preload_worker=ConfigureHelper.max_n_workers):
        super(TestDataset, self).__init__(data_root=data_root,
                                          split="test",
                                          preload_dataset=preload_dataset,
                                          ret_dsize=ret_dsize,
                                          n_preload_worker=n_preload_worker)
