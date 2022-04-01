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
from utils.TorchHelper import TorchHelper


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
    parser.add_argument("--gpu_id", type=int, default=-1, help="The id of gpu to be used. -1 for CPU only.")

    parser.add_argument("--n_epoch", type=int, required=True, help="Num of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--data_root",
                        type=str,
                        default="work_space/data",
                        help="Data root for storing Oxford Pet Data. "
                             "Files will be downloaded to the folder if data is not found.")
    parser.add_argument("--n_worker",
                        type=int,
                        default=ConfigureHelper.max_n_workers,
                        help="Num of worker for multi-processing")
    parser.add_argument("--preload_dataset",
                        type=TypeHelper.str2bool,
                        default=False,
                        help="True for preloading dataset into memory fastern training if large memory available.")


    parser.add_argument("--log_visualization_every_n_epoch", type=int, default=1)
    parser.add_argument("--save_weights_every_n_epoch", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    opt = parser.parse_args()
    opt.work_space_dir = OSHelper.format_path(opt.work_space_dir)
    opt.data_root = OSHelper.format_path(opt.data_root)

    if opt.gpu_id >= 0:
        if not torch.cuda.is_available():
            torch.cuda.init()
            raise RuntimeError(f"GPU {opt.gpu_id} is not available.")
        print(f"Running with GPU {opt.gpu_id}.")
    else:
        print(f"Running with CPU.")

    ConfigureHelper.set_seed(opt.seed)
    OSHelper.mkdirs(opt.work_space_dir)

    image_dsize = ContainerHelper.to_tuple(224)

    device = torch.device(opt.gpu_id if opt.gpu_id >= 0 else "cpu")
    if opt.gpu_id >= 0:
        print(torch.cuda.memory_summary(device))

    model = UNet(n_class=len(LABEL_NAME_DICT)).to(device)

    # Preparing datasets
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
    epoch = 1
    for epoch in range(epoch, opt.n_epoch):
        print("\nEpoch {} ({})".format(epoch, datetime.now()))
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Train one epoch
        epoch_loss = 0
        for image, label in tqdm(training_dataloader,
                                 total=len(training_dataloader),
                                 desc=f"Training"):
            # Move data to GPU if GPU is available
            image, label = image.to(device), label.to(device)

            # Calculate loss
            with torch.cuda.amp.autocast():
                pred_logits = model(image)
                loss = criter(pred_logits, label)

            # Standard steps for backpropagation
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # collect batch loss
            epoch_loss += loss
        # Log average loss
        epoch_loss /= len(training_dataloader)
        print(f"loss: {epoch_loss}")
        tb_writer.add_scalar("train_loss", epoch_loss, epoch)

        # Test one epoch
        # Prepare test result container
        test_dc = {label: 0 for label in LABEL_NAME_DICT}
        test_assd = {label: 0 for label in LABEL_NAME_DICT}
        sample_count = 0

        # Stop recording gradient for faster inference
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader,
                                       total=len(test_dataloader),
                                       desc="Testing"):
                images = images.to(device)
                pred_labels = torch.argmax(model(images), dim=1)  # (B, H, W)

                # Move data to CPU to be converted to numpy object
                labels = labels.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()

                # Iterate over Sample
                B = images.shape[0]  # Batch size
                for i in range(B):
                    label = labels[i]  # (H, W)
                    pred_label = pred_labels[i]  # (H, W)
                    for class_id in LABEL_NAME_DICT:
                        binary_label = label == class_id
                        binary_pred_label = pred_label == class_id

                        # calculate DC and ASSD
                        dc = EvaluationHelper.dc(binary_label, binary_pred_label)
                        try:
                            assd = EvaluationHelper.assd(binary_label, binary_pred_label)
                        except RuntimeError:
                            # In case of all-zero sample
                            binary_label[0, 0] = 1
                            binary_pred_label[-1, -1] = 1
                            assd = EvaluationHelper.assd(binary_label, binary_pred_label)
                        # collect evaluation results
                        test_dc[class_id] += dc
                        test_assd[class_id] += assd
                sample_count += 1
        # Average evaluation results
        for key in LABEL_NAME_DICT:
            test_dc[key] /= sample_count
            test_assd[key] /= sample_count
        test_dc = {LABEL_NAME_DICT[key]: val for key, val in test_dc.items()}
        test_assd = {LABEL_NAME_DICT[key]: val for key, val in test_assd.items()}
        test_dc["Mean"] = sum(test_dc.values()) / len(test_dc)
        test_assd["Mean"] = sum(test_assd.values()) / len(test_assd)

        # Log evaluation results
        msg = "Test DC:"
        for key, val in test_dc.items():
            msg += f" {key}({val:.3f})"
        print(msg)
        msg = "Test ASSD: "
        for key, val in test_assd.items():
            msg += f" {key}({val:.3f})"
        tb_writer.add_scalars("Test DC", test_dc, epoch)
        tb_writer.add_scalars("Test ASSD", test_assd, epoch)

        # Log visualization
        if epoch % opt.log_visualization_every_n_epoch == 0:
            with torch.no_grad():
                for images, labels in visualization_dataloader:
                    images = images.to(device)  # (B, 3, H, W)
                    pred_labels = torch.argmax(model(images), dim=1).cpu().numpy()  # (B, H, W)

                    image = images[0].permute(1, 2, 0).cpu().numpy()

                    # Denormalization
                    image = image * VisualizationDataset.NORMALIZATION_STD + VisualizationDataset.NORMALIZATION_MEAN
                    image = image.clip(0., 1.)

                    # Convert to standard data range [0, 255]
                    image = (image * 255.).astype(np.uint8)

                    label = labels[0].cpu().numpy().astype(np.uint8).squeeze()
                    label = cv2.cvtColor(cv2.applyColorMap(label, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                    pred_label = pred_labels[0].cpu().numpy().astype(np.uint8).squeeze()
                    pred_label = cv2.cvtColor(cv2.applyColorMap(pred_label, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

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

        if epoch & opt.save_weights_every_n_epoch == 0:
            TorchHelper.save_network(model, OSHelper.path_join(opt.work_space_dir, f"net_{epoch}.pth"))
    TorchHelper.save_network(model, OSHelper.path_join(opt.work_space_dir, f"net_{epoch - 1}.pth"))


    pass


if __name__ == '__main__':
    main()
