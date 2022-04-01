#  Copyright (c) 2021-2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List, Union
from abc import ABC
import warnings


class ImageHelper(ABC):

    @staticmethod
    def resize(image: np.ndarray, dsize: Union[tuple[int, int], list[int, int]]) -> np.ndarray:
        _, _, = image.shape[: 2]
        image = cv2.resize(image, dsize)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        return image

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


    @staticmethod
    def standardize(image: np.ndarray or torch.Tensor,
                    mean: Union[int, float, None] = None,
                    std: Union[int, float, None] = None) -> np.float or torch.Tensor:
        if mean is None:
            mean = image.mean()
        if std is None:
            std = image.std()
        return (image - mean) / std

    @staticmethod
    def denormal(image: Union[np.ndarray, torch.Tensor],
                 ret_min_val: float = 0.,
                 ret_max_val: float = 255.) -> Union[np.ndarray, torch.Tensor]:
        """
        :param image: Normalized image with range [-1, 1]
        :param ret_min_val: min value of returned space
        :param ret_max_val: max value of returned space
        :return: denormalized image
        """
        if image.min() < 0. or image.max() > 1.:
            raise RuntimeError(f"Unexpected data range: {image.min()} {image.max()}")
        return image * (ret_max_val - ret_min_val) + ret_min_val

    @staticmethod
    def normalize(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
            convert image [0, 255] -> [0., 1.]
        :param image: Target image
        :return: Normalized image
        """
        if image.min() < 0 or image.max() > 255.:
            raise RuntimeError(f"Unexpected data range: {image.min()} {image.max()}")
        return image / 255.

    @staticmethod
    def intensity_scaling(x: np.ndarray or torch.Tensor) -> np.ndarray or torch.Tensor:
        return ImageHelper.min_max_scale(x) * 255.

    @staticmethod
    def resize_deep_channel(img: np.ndarray, dsize: tuple[int, int], splitting_size: int = 3) -> np.ndarray:
        W, H = dsize
        C = img.shape[-1]
        ret = np.zeros((H, W, C), dtype=img.dtype)
        num_of_splitting = C // splitting_size
        if C % splitting_size != 0:
            num_of_splitting += 1
        for i in range(num_of_splitting):
            start_i = i * splitting_size
            end_i = start_i + splitting_size
            resized = cv2.resize(img[:, :, start_i: end_i], dsize=dsize)
            if resized.ndim < 3:
                resized = np.expand_dims(resized, -1)
            ret[:, :, start_i: end_i] = resized
        return ret

    @staticmethod
    def dense_map_to_logits(dense_map: Union[np.ndarray, torch.Tensor],
                            n_class: int = None) -> Union[np.ndarray, torch.Tensor]:
        """

        :param n_class:
        :param dense_map: (N, H, W) or (H, W) np.uint8
        :return: (N, H, W, C) or (H, W, C)
        """
        if n_class is None:
            n_class = 0
        assert n_class >= 0
        num_of_labels = dense_map.max() + 1 if n_class == 0 else n_class
        logits_shape = dense_map.shape + (num_of_labels,)
        if isinstance(dense_map, np.ndarray):
            logits = np.zeros(logits_shape, dtype=float)
        else:
            logits = torch.zeros(*logits_shape, dtype=torch.float, device=dense_map.device)
        for i in range(num_of_labels):
            logits[..., i] = dense_map == i
        return logits

    @staticmethod
    def blend(image1: Union[np.ndarray, torch.Tensor],
              image2: Union[np.ndarray, torch.Tensor],
              alpha: float) -> Union[np.ndarray, torch.Tensor]:
        """
        Creates a new image by interpolating between two input images, using
        a constant alpha.::

        out = image1 * (1.0 - alpha) + image2 * alpha
        :param image1: > 0.
        :param image2: > 0.
        :param alpha:
        :return:
        """
        assert image1.min() >= 0. and image2.min() >= 0.
        return image1 * (1. - alpha) + image2 * alpha

    @staticmethod
    def contrast(image: Union[np.ndarray, torch.Tensor],
                 factor: float) -> Union[np.ndarray, torch.Tensor]:
        """
         An enhancement factor of 0.0 gives a solid grey image. A factor of 1.0 gives the original image.
        :param image: > 0.
        :param factor: > 0.
        :return:
        """
        if isinstance(image, np.ndarray):
            image1 = np.full_like(image, image.mean())
        else:
            image1 = torch.full_like(image, image.mean())
        return ImageHelper.blend(image1=image1, image2=image, alpha=factor)

    @staticmethod
    def brightness(image: Union[np.ndarray, torch.Tensor],
                   factor: float) -> Union[np.ndarray, torch.Tensor]:
        """
        factor of 0.0 gives a black image. A factor of 1.0 gives the original image.
        :param image: > 0.
        :param factor:
        :return:
        """
        if isinstance(image, np.ndarray):
            image1 = np.full_like(image, 0)
        else:
            image1 = torch.full_like(image, 0)
        return ImageHelper.blend(image1=image1, image2=image, alpha=factor)

    @staticmethod
    def center_cropping(img: Union[np.ndarray, torch.Tensor], w_to_h_ratio=1) -> Union[np.ndarray, torch.Tensor]:
        """
        :param img: (H, W, ...)
        :param w_to_h_ratio:
        :return:
        """
        H, W = img.shape[: 2]
        if W / H < w_to_h_ratio:
            new_H = W / w_to_h_ratio
            y = round((H - new_H) / 2)
            return img[y: y + round(new_H)]
        elif W / H > w_to_h_ratio:
            new_W = H * w_to_h_ratio
            x = round((W - new_W) / 2)
            return img[:, x: x + round(new_W)]
        return img


