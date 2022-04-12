#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
import numpy as np
from torch.utils.data import Dataset
from abc import ABC
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
import os
import os.path
import pathlib
from PIL import Image
from utils.ConfigureHelper import ConfigureHelper
from MultiProcessingHelper import MultiProcessingHelper
from typing import Union, AnyStr
from utils.ImageHelper import ImageHelper
from utils.ContainerHelper import ContainerHelper
import cv2


class BaseDataset(Dataset, ABC):
    """
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/oxford_iiit_pet.html
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
         "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
         "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]

    LABEL_NAME_DICT = {0: "Foreground", 1: "Background", 2: "Not-classified"}

    def __init__(self,
                 data_root: str,
                 split: str,
                 preload_dataset: bool,
                 ret_dsize: Union[int, tuple[int, int]],
                 n_preload_worker=ConfigureHelper.max_n_workers):
        super(BaseDataset, self).__init__()
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        self._base_folder = pathlib.Path(data_root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"
        self.__preload_dataset = preload_dataset
        self.__preloaded = False
        self.__image_load_func = self._load_image
        self.__seg_load_func = self._load_seg
        self._ret_dsize = ContainerHelper.to_tuple(ret_dsize)

        if not self._check_exists():
            self._download()
        self._image_ids = self.read_image_ids(self._anns_folder / f"{self._split}.txt")
        self._image_id_idx_dict = {image_id: i for i, image_id in enumerate(self._image_ids)}


        self._images = [str(self._images_folder / f"{image_id}.jpg") for image_id in self._image_ids]
        self._segs = [str(self._segs_folder / f"{image_id}.png") for image_id in self._image_ids]

        if self.__preload_dataset:
            mph = MultiProcessingHelper()
            self._images = mph.run(args=[(image,) for image in self._images],
                                   func=self._load_image,
                                   process_bar=True,
                                   n_workers=n_preload_worker,
                                   desc=f"Pre-loading {split} images")
            self._segs = mph.run(args=[(seg,) for seg in self._segs],
                                 func=self._load_seg,
                                 process_bar=True,
                                 n_workers=n_preload_worker,
                                 desc=f"Pre-loading {split} labels")
            self.__image_load_func = self._identity
            self.__seg_load_func = self._identity

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        image = self.__image_load_func(self._images[idx])
        seg = self.__seg_load_func(self._segs[idx])
        image, seg = self._augment(image=image, seg=seg)
        image = cv2.resize(image, dsize=self._ret_dsize)
        seg = cv2.resize(seg, dsize=self._ret_dsize, interpolation=cv2.INTER_NEAREST)
        return image.transpose((2, 0, 1)).astype(np.float32), seg.astype(int), self._image_ids[idx]

    def get_item_by_id(self, image_id: str) -> tuple[np.ndarray, np.ndarray, str]:
        return self.__getitem__(self._image_id_idx_dict[image_id])

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return
        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)

    @classmethod
    def _load_image(cls, path) -> np.ndarray:
        ret = np.array(Image.open(path).convert("RGB")).astype(float) / 255.
        ret = (ret - cls.NORMALIZATION_MEAN) / cls.NORMALIZATION_STD
        return ret

    @staticmethod
    def _load_seg(path) -> np.ndarray:
        return np.array(Image.open(path)) - 1  # 1, 2, 3 to 0, 1, 2

    @staticmethod
    def read_image_ids(path) -> list[str]:
        image_ids = []
        with open(path) as file:
            for i, line in enumerate(file):
                image_id, *_ = line.strip().split()
                image_ids.append(image_id)
        return image_ids

    @staticmethod
    def _identity(x):
        return x

    def _augment(self, image, seg):
        return image, seg
