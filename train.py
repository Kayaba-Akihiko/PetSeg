#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import argparse

import numpy as np

from utils.ConfigureHelper import ConfigureHelper
import torch
import sys
from model import UNet
from Dataset import TrainingDataset, TestDataset, VisualizationDataset
from torch.utils.data import DataLoader
from utils.OSHelper import OSHelper
from utils.ContainerHelper import ContainerHelper
from utils.TypeHelper import TypeHelper
from utils.ConfigureHelper import ConfigureHelper
from torch.optim import Adam
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from utils.EvaluationHelper import EvaluationHelper
import matplotlib.pyplot as plt
import cv2


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(False)
if sys.platform.startswith("linux"):
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

LABEL_NAME_DICT = {0: "Foreground", 1: "Background", 2: "Not-classified"}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_space_dir", type=str, default="work_space/default")
    parser.add_argument("--n_worker", type=int, default=ConfigureHelper.max_n_workers)
    parser.add_argument("--data_root", type=str, default="work_space/data")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--preload_dataset", type=TypeHelper.str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_visualization_every_n_epoch", type=int, default=1)

    parser.add_argument("--n_epoch", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=0)

    opt = parser.parse_args()
    opt.work_space_dir = OSHelper.format_path(opt.work_space_dir)
    opt.data_root = OSHelper.format_path(opt.data_root)

    ConfigureHelper.set_seed(opt.seed)
    OSHelper.mkdirs(opt.work_space_dir)

    image_dsize = ContainerHelper.to_tuple(224)

    device = torch.device(opt.gpu_id if opt.gpu_id >= 0 else "cpu")

    model = UNet(n_class=len(LABEL_NAME_DICT)).to(device)

    training_dataloader = TrainingDataset(data_root=opt.data_root,
                                          preload_dataset=opt.preload_dataset,
                                          ret_dsize=image_dsize,
                                          n_preload_worker=opt.n_worker)
    training_dataloader = DataLoader(training_dataloader,
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     num_workers=opt.n_worker,
                                     pin_memory=sys.platform.startswith("linux"))
    test_dataloader = TestDataset(data_root=opt.data_root,
                                  preload_dataset=opt.preload_dataset,
                                  ret_dsize=image_dsize,
                                  n_preload_worker=opt.n_worker)
    test_dataloader = DataLoader(test_dataloader,
                                 batch_size=opt.batch_size,
                                 num_workers=opt.n_worker,
                                 pin_memory=sys.platform.startswith("linux"))
    visualization_dataloader = VisualizationDataset(data_root=opt.data_root,
                                                    ret_dsize=image_dsize)
    visualization_dataloader = DataLoader(visualization_dataloader)

    optimizer = Adam(model.decoder.parameters(), lr=opt.lr)
    grad_scaler = torch.cuda.amp.GradScaler()
    criter = torch.nn.CrossEntropyLoss().to(device)

    tb_writer = SummaryWriter(log_dir=OSHelper.path_join(opt.work_space_dir, "tb_log"))
    for epoch in range(1, opt.n_epoch):
        print("\nEpoch {} ({})".format(epoch, datetime.now()))
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        epoch_loss = 0
        for image, label in tqdm(training_dataloader,
                                 total=len(training_dataloader),
                                 desc=f"Training",
                                 # mininterval=ConfigureHelper.TQDM_INTERVAL[0],
                                 # maxinterval=ConfigureHelper.TQDM_INTERVAL[1]
                                 ):
            image, label = image.to(device), label.to(device)
            with torch.cuda.amp.autocast():
                pred_logits = model(image)
                loss = criter(pred_logits, label)
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            epoch_loss += loss
        epoch_loss /= len(training_dataloader)
        print(f"loss: {epoch_loss}")
        tb_writer.add_scalar("train_loss", epoch_loss, epoch)

        test_dc = {label: 0 for label in LABEL_NAME_DICT}
        test_assd = {label: 0 for label in LABEL_NAME_DICT}
        sample_count = 0
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader,
                                       total=len(test_dataloader),
                                       desc="Testing",
                                       # mininterval=ConfigureHelper.TQDM_INTERVAL[0],
                                       # maxinterval=ConfigureHelper.TQDM_INTERVAL[1]
                                       ):
                images = images.to(device)
                pred_labels = torch.argmax(model(images), dim=1)  # (B, H, W)
                B = images.shape[0]

                labels = labels.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                for i in range(B):
                    label = labels[i]  # (H, W)
                    pred_label = pred_labels[i]  # (H, W)
                    for class_id in LABEL_NAME_DICT:
                        binary_label = label == class_id
                        binary_pred_label = pred_label == class_id
                        dc = EvaluationHelper.dc(binary_label, binary_pred_label)
                        try:
                            assd = EvaluationHelper.assd(binary_label, binary_pred_label)
                        except RuntimeError:
                            assd = np.inf
                        test_dc[class_id] += dc
                        test_assd[class_id] += assd
                sample_count += 1
        for key in LABEL_NAME_DICT:
            test_dc[key] /= sample_count
            test_assd[key] /= sample_count
        test_dc = {LABEL_NAME_DICT[key]: val for key, val in test_dc.items()}
        test_assd = {LABEL_NAME_DICT[key]: val for key, val in test_assd.items()}
        test_dc["Mean"] = sum(test_dc.values()) / len(test_dc)
        test_assd["Mean"] = sum(test_assd.values()) / len(test_assd)
        msg = "Test DC:"
        for key, val in test_dc.items():
            msg += f" {key}({val:.3f})"
        print(msg)
        msg = "Test ASSD: "
        for key, val in test_assd.items():
            msg += f" {key}({val:.3f})"
        tb_writer.add_scalars("Test DC", test_dc, epoch)
        tb_writer.add_scalars("Test ASSD", test_assd, epoch)

        if epoch % opt.log_visualization_every_n_epoch == 0:
            with torch.no_grad():
                for images, labels in visualization_dataloader:
                    images = images.to(device)  # (B, 3, H, W)
                    pred_labels = torch.argmax(model(images), dim=1).cpu().numpy()  # (B, H, W)

                    image = images[0].permute(1, 2, 0).cpu().numpy()
                    image = image * VisualizationDataset.NORMALIZATION_STD + VisualizationDataset.NORMALIZATION_MEAN
                    image = image.clamp(0., 1.)
                    image = (image * 255.).astype(np.uint8)

                    label = labels[0].cpu().numpy()
                    label = cv2.applyColorMap(label, cv2.COLORMAP_VIRIDIS)
                    pred_label = pred_labels[0].cpu().numpy()
                    pred_label = cv2.applyColorMap(pred_label, cv2.COLORMAP_VIRIDIS)

                    titles = ['Input Image', 'True Mask', 'Predicted Mask']
                    display_list = [image, label, pred_label]

                    for title, image in zip(titles, display_list):
                        tb_writer.add_image(tag=f"e{epoch} {title}", img_tensor=image, dataformats="HWC")
                    #
                    #
                    #
                    # plt.clf()
                    # fig = plt.figure(figsize=(15, 15))
                    #
                    # for i in range(len(display_list)):
                    #     plt.subplot(1, len(display_list), i + 1)
                    #     plt.title(title[i])
                    #     plt.imshow(display_list[i])
                    #     plt.axis('off')
                    # fig.canvas.draw()
                    # tb_writer.add_image(tag=f"Epoch {epoch}",
                    #                     img_tensor=np.array(fig.canvas.renderer.buffer_rgba()),
                    #                     global_step=epoch,
                    #                     dataformats="HWC")
                    # plt.close(fig)
                    # plt.close()

                    break

    pass


if __name__ == '__main__':
    main()