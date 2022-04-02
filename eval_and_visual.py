#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import argparse

import cv2

from utils.ConfigureHelper import ConfigureHelper
from utils.OSHelper import OSHelper
import torch
from utils.ContainerHelper import ContainerHelper
from Dataset import TestDataset
from model import UNet
from utils.TorchHelper import TorchHelper
import sys
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils.EvaluationHelper import EvaluationHelper
from MultiProcessingHelper import MultiProcessingHelper
import itertools

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(False)
if sys.platform.startswith("linux"):
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_space_dir",
                        type=str,
                        default="work_space/default",
                        help="Work space directory for this running, "
                             "containing the pretrained weights of model to be evaluated. "
                             "Output will also be save to this directory")
    parser.add_argument("--gpu_id", type=int, default=-1, help="The id of gpu to be used. -1 for CPU only.")
    parser.add_argument("--pretrain_loading_epoch", type=int, required=True, help="The epoch weights to load")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_worker",
                        type=int,
                        default=ConfigureHelper.max_n_workers,
                        help="Num of worker for multi-processing")
    parser.add_argument("--data_root",
                        type=str,
                        default="work_space/data",
                        help="Data root for storing Oxford Pet Data. "
                             "Files will be downloaded to the folder if data is not found.")

    opt = parser.parse_args()
    opt.work_space_dir = OSHelper.format_path(opt.work_space_dir)
    opt.data_root = OSHelper.format_path(opt.data_root)

    image_dsize = ContainerHelper.to_tuple(224)

    inference_save_dir = OSHelper.path_join(opt.work_space_dir, f"inference_{opt.pretrain_loading_epoch}")
    OSHelper.mkdirs(inference_save_dir)
    # __inference(opt, image_dsize, inference_save_dir)

    eval_save_dir = OSHelper.path_join(opt.work_space_dir, f"eval_{opt.pretrain_loading_epoch}")
    OSHelper.mkdirs(eval_save_dir)
    image_ids = TestDataset.read_image_ids(OSHelper.path_join(opt.data_root,
                                                              "oxford-iiit-pet",
                                                              "annotations",
                                                              "test.txt"))
    args = []
    for image_id in image_ids:
        pred_path = OSHelper.path_join(inference_save_dir, f"{image_id}.png")
        target_path = OSHelper.path_join(opt.data_root, "oxford-iiit-pet", "annotations", "trimaps", f"{image_id}.png")
        args.append((pred_path, target_path, image_id, image_dsize))

    eval_df = MultiProcessingHelper().run(args=args,
                                          func=_load_and_eval,
                                          n_workers=opt.n_worker,
                                          process_bar=True,
                                          desc="Evaluating")
    eval_df = list(itertools.chain(*eval_df))
    eval_df = pd.DataFrame(eval_df, index=range(len(eval_df)))
    eval_df.to_excel(OSHelper.path_join(eval_save_dir, "eval.xlsx"))


def __inference(opt, image_dsize, inference_save_dir) -> None:
    if opt.gpu_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU {opt.gpu_id} is not available.")
        else:
            torch.cuda.init()
        print(f"Running with GPU {opt.gpu_id}.")
    else:
        print(f"Running with CPU.")
    device = torch.device(opt.gpu_id if opt.gpu_id >= 0 else "cpu")

    model = UNet(n_class=len(TestDataset.LABEL_NAME_DICT)).to(device)

    model_load_path = OSHelper.path_join(opt.work_space_dir, f"net_{opt.pretrain_loading_epoch}.pth")
    if not OSHelper.path_exists(model_load_path):
        raise RuntimeError(f"Model weights not found at {model_load_path} .")
    TorchHelper.load_network_by_path(net=model, path=model_load_path, device=device)

    test_dataloader = TestDataset(data_root=opt.data_root,
                                  preload_dataset=False,
                                  ret_dsize=image_dsize,
                                  n_preload_worker=opt.n_worker)
    test_dataloader = DataLoader(test_dataloader,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.n_worker,
                                 pin_memory=sys.platform.startswith("linux"))

    with torch.no_grad():
        for images, labels, image_ids in tqdm(test_dataloader, total=len(test_dataloader), desc="Inferencing"):
            images = images.to(device)
            pred_labels = torch.argmax(model(images), dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)
            pred_labels += 1  # [0, 1, 2] to [1, 2, 3]

            B = len(image_ids)
            for i in range(B):
                image_id = image_ids[i]
                pred_label = pred_labels[i]
                Image.fromarray(pred_label).save(OSHelper.path_join(inference_save_dir, f"{image_id}.png"))


def _load_and_eval(pred_path, target_path, image_id, image_dsize) -> list[dict]:
    pred_seg = np.array(Image.open(pred_path)) - 1
    target_seg = np.array(Image.open(target_path)) - 1

    target_seg = cv2.resize(target_seg, image_dsize, interpolation=cv2.INTER_NEAREST)

    data = []
    mean_dc = 0
    mean_assd = 0
    for class_id in TestDataset.LABEL_NAME_DICT:
        binary_label = target_seg == class_id
        binary_pred_label = pred_seg == class_id
        # calculate DC and ASSD
        dc = EvaluationHelper.dc(binary_label, binary_pred_label)
        try:
            assd = EvaluationHelper.assd(binary_label, binary_pred_label)
        except RuntimeError:
            # In case of all-zero sample
            binary_label[0, 0] = 1
            binary_pred_label[-1, -1] = 1
            assd = EvaluationHelper.assd(binary_label, binary_pred_label)
        data.append({"image_id": image_id, "class": TestDataset.LABEL_NAME_DICT[class_id], "metric": "DC", "value": dc})
        data.append({"image_id": image_id, "class": TestDataset.LABEL_NAME_DICT[class_id], "metric": "ASSD", "value": assd})
        mean_dc += dc
        mean_assd += assd
    mean_dc /= len(TestDataset.LABEL_NAME_DICT)
    mean_assd /= len(TestDataset.LABEL_NAME_DICT)
    data.append({"image_id": image_id, "class": "mean", "metric": "DC", "value": mean_dc})
    data.append({"image_id": image_id, "class": "mean", "metric": "ASSD", "value": mean_assd})
    return data


if __name__ == '__main__':
    main()