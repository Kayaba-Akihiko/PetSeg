#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from collections.abc import Sequence
from abc import ABC

class ContainerHelper(ABC):

    @staticmethod
    def to_tuple(x):
        if not isinstance(x, Sequence):
            return x, x
        return x
