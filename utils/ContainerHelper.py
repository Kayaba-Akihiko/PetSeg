#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import numpy as np
from operator import itemgetter
import random
from collections.abc import Sequence
from typing import Union, Any


class ContainerHelper:

    @staticmethod
    def select_item_by_indexes(container: Union[np.ndarray, Sequence[Any]],
                               indexes: Union[np.ndarray, Sequence[int]]) -> Union[list[Any], np.ndarray]:
        if isinstance(container, np.ndarray) and isinstance(indexes, np.ndarray):
            assert container.ndim == 1 and indexes.ndim == 1, f"{container.ndim}, {indexes.ndim}"
            return container[indexes]

        assert isinstance(container, Sequence) and isinstance(indexes, Sequence), f"{type(container)}, {type(indexes)}"
        if len(indexes) == 0:
            return []
        return [container[indexes[0]]] if len(indexes) == 1 else list(itemgetter(*indexes)(container))

    @classmethod
    def sample_n_items(cls,
                       container: Union[np.ndarray, Sequence[Any]],
                       n_samples: Union[int, Sequence[int]]) -> tuple[list[Any, ...], list[int, ...]]:
        if isinstance(container, np.ndarray):
            assert container.ndim == 1, f"{container.ndim}"
        else:
            assert isinstance(container, Sequence), f"{type(container)}"
        sample_indexes = n_samples

        if not isinstance(n_samples, Sequence):
            if n_samples == 0:
                return [], []
            assert 0 < n_samples <= len(container), f"{n_samples}, {len(container)}"
            sample_indexes = random.sample([i for i in range(len(container))], n_samples)
        return cls.select_item_by_indexes(container, sample_indexes), sample_indexes

    @staticmethod
    def item_to_index(items: Any, container: Sequence) -> list[int, ...]:
        assert isinstance(container, Sequence), f"{type(container)}"
        if not isinstance(items, Sequence):
            items = [items]
        return [container.index(item) for item in items]

    @staticmethod
    def to_tuple(x):
        if not isinstance(x, Sequence):
            return x, x
        return x
