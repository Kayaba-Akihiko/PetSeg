#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import cv2
import numpy as np
import torch
from typing import Union
from abc import ABC


class ImageHelper(ABC):

    @staticmethod
    def min_max_scale(x: np.ndarray or torch.Tensor,
                      return_zero_for_except: bool,
                      min_val: Union[int, float, None] = None,
                      max_val: Union[int, float, None] = None) -> np.ndarray or torch.Tensor:
        """

        :param x:
        :param return_zero_for_except:
        :param min_val:
        :param max_val:
        :return:
            [0., 1.]
        """
        if min_val is None and max_val is None:
            min_val = x.min()
            max_val = x.max()
        try:
            if min_val is None and max_val is not None or min_val is not None and max_val is None:
                raise RuntimeError("Min_val or max_val is None. min_val {}, max_val {}.".format(min_val, max_val))
            if min_val > max_val:
                raise RuntimeError("Unsupported min and max. min_val {}, max_val {}.".format(min_val, max_val))
            if not (x.min() >= min_val and x.max() <= max_val):
                raise RuntimeError(f"Unexpected data range: {x.min()} {min_val} {x.max()} {max_val}")
            if max_val == min_val:
                raise RuntimeError(f"Constant value:", {min_val, max_val})
            return (x - min_val) / (max_val - min_val)
        except RuntimeError as e:
            if return_zero_for_except:
                if isinstance(x, np.ndarray):
                    return np.zeros_like(x)
                if isinstance(x, torch.Tensor):
                    return torch.zeros_like(x, device=x.device)
                raise RuntimeError(f"Unsupported type {type(x)}.")
            raise e

    @classmethod
    def apply_colormap_to_dense_map(cls, dense_mape, min_class_id=0, max_class_id=255, color_map=cv2.COLORMAP_VIRIDIS):
        dense_map = cls.min_max_scale(dense_mape.squeeze().astype(float),
                                      False,
                                      min_val=min_class_id,
                                      max_val=max_class_id) * 255.
        dense_map = cv2.applyColorMap(dense_map.astype(np.uint8), color_map)
        return dense_map
