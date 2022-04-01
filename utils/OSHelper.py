#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import os
import os.path as osp
from os import makedirs, scandir
from typing import Union, Optional
import re
from collections.abc import Sequence, Callable
from typing import AnyStr


class OSHelper:

    @staticmethod
    def format_path(path: AnyStr, src_sep='\\', ret_ep='/') -> AnyStr:
        return path.replace(src_sep, ret_ep)

    @staticmethod
    def path_join(*paths: AnyStr) -> AnyStr:
        return osp.join(*paths)

    @staticmethod
    def path_dirname(path: str) -> AnyStr:
        return osp.dirname(path)

    @staticmethod
    def mkdirs(paths: Union[Union[list[AnyStr, ...], tuple[AnyStr, ...]], AnyStr]):
        if (isinstance(paths, list) or isinstance(paths, tuple)) and not isinstance(paths, str):
            for path in paths:
                makedirs(path, exist_ok=True)
        else:
            makedirs(paths, exist_ok=True)

    @classmethod
    def scan_dirs_for_folder(cls, paths: Union[str, Sequence[str]],
                             name_re_pattern: Optional[re.Pattern] = None) -> list[os.DirEntry]:
        return cls.__scan_dirs(paths=paths, allow_fun=cls.__allow_dir, name_re_pattern=name_re_pattern)

    @classmethod
    def scan_dirs_for_file(cls, paths: Union[str, Sequence[str]],
                           name_re_pattern: Optional[re.Pattern] = None) -> list[os.DirEntry]:
        return cls.__scan_dirs(paths=paths, allow_fun=cls.__allow_file, name_re_pattern=name_re_pattern)

    @staticmethod
    def __allow_dir(entry: os.DirEntry) -> bool:
        return entry.is_dir()

    @staticmethod
    def __allow_file(entry: os.DirEntry) -> bool:
        return entry.is_file()

    @staticmethod
    def __scan_dirs(paths: Union[str, Sequence[str]],
                    allow_fun: Callable[[os.DirEntry], bool],
                    name_re_pattern: Optional[re.Pattern] = None) -> list[os.DirEntry]:
        if isinstance(paths, str):
            paths = [paths]
        if name_re_pattern is None:
            name_re_pattern = re.compile(",*")
        ret = []
        for path in paths:
            with os.scandir(path) as it:
                entry: os.DirEntry
                ret.extend([entry for entry in it
                            if allow_fun(entry) and name_re_pattern.match(entry.name) is not None])
        return ret