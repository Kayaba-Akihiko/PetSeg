#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from collections.abc import Iterable


class StringHelper:

    @staticmethod
    def str_lt(a: str, b: str) -> bool:
        return a < b if len(a) == len(b) else len(a) < len(b)

    @staticmethod
    def string_item_list_matching(item: str, container: Iterable[str], case_avoid=True) -> bool:
        temp_container = list(map(str.lower, container)) if case_avoid else container
        temp_item = item.lower() if case_avoid else item
        return temp_item in temp_container
