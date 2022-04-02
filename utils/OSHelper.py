#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import os.path as osp
from os import makedirs
from typing import Union
from typing import AnyStr
from abc import ABC


class OSHelper(ABC):

    @staticmethod
    def format_path(path: AnyStr, src_sep='\\', ret_ep='/') -> AnyStr:
        return path.replace(src_sep, ret_ep)

    @staticmethod
    def path_join(*paths: AnyStr) -> AnyStr:
        return osp.join(*paths)

    @staticmethod
    def path_dirname(path: AnyStr) -> AnyStr:
        return osp.dirname(path)

    @staticmethod
    def path_exists(path: AnyStr) -> bool:
        return osp.exists(path)

    @staticmethod
    def mkdirs(paths: Union[Union[list[AnyStr, ...], tuple[AnyStr, ...]], AnyStr]):
        if (isinstance(paths, list) or isinstance(paths, tuple)) and not isinstance(paths, str):
            for path in paths:
                makedirs(path, exist_ok=True)
        else:
            makedirs(paths, exist_ok=True)