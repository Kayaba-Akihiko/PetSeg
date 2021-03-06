#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import argparse

import numpy as np

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
import cv2
from utils.TorchHelper import TorchHelper
from utils.ImageHelper import ImageHelper


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_space_dir",
                        type=str,
                        default="work_space/default",
                        help="Work space directory for this running. Output will be save to this directory")
    parser.add_argument("--gpu_id", type=int, default=-1, help="The id of gpu to be used. -1 for CPU only.")

    parser.add_argument("--n_epoch", type=int, required=True, help="Num of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument("--n_worker",
                        type=int,
                        default=ConfigureHelper.max_n_workers,
                        help="Num of worker for multi-processing")

    parser.add_argument("--data_root",
                        type=str,
                        default="work_space/data",
                        help="Data root for storing Oxford Pet Data. "
                             "Files will be downloaded to the folder if data is not found.")
    parser.add_argument("--preload_dataset",
                        type=TypeHelper.str2bool,
                        default=False,
                        help="True for preloading dataset into memory fasten training if large memory available.")

    parser.add_argument("--log_visualization_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_model_histogram_every_n_epoch", type=int, default=1)
    parser.add_argument("--save_weights_every_n_epoch", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    opt = parser.parse_args()
    opt.work_space_dir = OSHelper.format_path(opt.work_space_dir)
    opt.data_root = OSHelper.format_path(opt.data_root)
    ConfigureHelper.set_seed(opt.seed)
    OSHelper.mkdirs(opt.work_space_dir)

    # Save setting
    opt_str = __serialize_option(opt, parser)
    print(opt_str)
    try:
        with open(OSHelper.path_join(opt.work_space_dir, "opt.txt"), 'wt') as opt_file:
            opt_file.write(opt_str)
            opt_file.write('\n')
    except PermissionError as error:
        print("permission error {}".format(error))
        pass

    if opt.gpu_id >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(f"GPU {opt.gpu_id} is not available.")
        else:
            torch.cuda.init()
        print(f"Running with GPU {opt.gpu_id}.")
    else:
        print(f"Running with CPU.")

    device = torch.device(opt.gpu_id if opt.gpu_id >= 0 else "cpu")
    image_dsize = ContainerHelper.to_tuple(224)

    model = UNet(n_class=len(TrainingDataset.LABEL_NAME_DICT)).to(device)
    with open(OSHelper.path_join(opt.work_space_dir, "net.txt"), 'wt') as opt_file:
        opt_file.write(str(model))
        opt_file.write('\n')

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
    # mph = MultiProcessingHelper()

    tb_writer = SummaryWriter(log_dir=OSHelper.path_join(opt.work_space_dir, "tb_log"))
    tb_writer.add_graph(model, torch.randn(2, 3, *image_dsize[:: -1], device=device))
    epoch = 1
    for epoch in range(epoch, opt.n_epoch + 1):
        print("\nEpoch {} ({})".format(epoch, datetime.now()))
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Train one epoch
        epoch_loss = 0
        for image, label, _ in tqdm(training_dataloader,
                                    total=len(training_dataloader),
                                    desc=f"Training"):
            # Move data to GPU if GPU is available
            image, label = image.to(device), label.to(device)

            # Calculate loss
            with torch.cuda.amp.autocast():  # Automatic Mixed Precision
                pred_logits = model(image)
                loss = criter(pred_logits, label)

            # Standard steps for backpropagation
            optimizer.zero_grad()
            # Automatic Mixed Precision
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
        test_dc = {label: 0 for label in TrainingDataset.LABEL_NAME_DICT}
        test_assd = {label: 0 for label in TrainingDataset.LABEL_NAME_DICT}
        sample_count = 0

        with torch.no_grad():  # Stop recording gradient for faster inference
            for images, labels, _ in tqdm(test_dataloader,
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
                    label = labels[i]
                    pred_label = pred_labels[i]
                    for class_id in TrainingDataset.LABEL_NAME_DICT:
                        binary_label = label == class_id
                        binary_pred_label = pred_label == class_id
                        dc, assd = EvaluationHelper.dc_and_assd(binary_pred_label, binary_label)
                        test_dc[class_id] += dc
                        test_assd[class_id] += assd

                sample_count += B
        # Average evaluation results
        for key in TrainingDataset.LABEL_NAME_DICT:
            test_dc[key] /= sample_count
            test_assd[key] /= sample_count
        test_dc = {TrainingDataset.LABEL_NAME_DICT[key]: val for key, val in test_dc.items()}
        test_assd = {TrainingDataset.LABEL_NAME_DICT[key]: val for key, val in test_assd.items()}
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
        print(msg)
        tb_writer.add_scalars("Test DC", test_dc, epoch)
        tb_writer.add_scalars("Test ASSD", test_assd, epoch)

        # Log visualization
        if epoch % opt.log_visualization_every_n_epoch == 0:
            with torch.no_grad():
                for images, labels, _ in visualization_dataloader:
                    images = images.to(device)  # (B, 3, H, W)
                    pred_labels = torch.argmax(model(images), dim=1)  # (B, H, W)

                    image = images[0].permute(1, 2, 0).cpu().numpy()
                    label = labels[0].cpu().numpy()
                    pred_label = pred_labels[0].cpu().numpy()

                    # Denormalization
                    image = image * VisualizationDataset.NORMALIZATION_STD + VisualizationDataset.NORMALIZATION_MEAN
                    image = image.clip(0., 1.)
                    # Convert to standard data range [0, 255]
                    image = (image * 255.).astype(np.uint8)

                    min_class_id = min(TrainingDataset.LABEL_NAME_DICT.keys())
                    max_class_id = max(TrainingDataset.LABEL_NAME_DICT.keys())
                    label = ImageHelper.apply_colormap_to_dense_map(label,
                                                                    min_class_id=min_class_id,
                                                                    max_class_id=max_class_id)
                    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                    pred_label = ImageHelper.apply_colormap_to_dense_map(pred_label,
                                                                         min_class_id=min_class_id,
                                                                         max_class_id=max_class_id)
                    pred_label = cv2.cvtColor(pred_label, cv2.COLOR_BGR2RGB)

                    titles = ['Input Image', 'True Mask', 'Predicted Mask']
                    display_list = [image, label, pred_label]

                    for title, image in zip(titles, display_list):
                        tb_writer.add_image(tag=f"e{epoch} {title}", img_tensor=image, dataformats="HWC")
                    break

        if epoch % opt.log_model_histogram_every_n_epoch == 0:
            for name, param in model.named_parameters():
                if name.startswith("decoder"):
                    tb_writer.add_histogram(name, param.clone().cpu().detach().numpy(), epoch)

        if epoch % opt.save_weights_every_n_epoch == 0:
            TorchHelper.save_network(model, OSHelper.path_join(opt.work_space_dir, f"net_{epoch}.pth"))

        if epoch == 1:
            if opt.gpu_id >= 0:
                print(torch.cuda.memory_summary(opt.gpu_id))
    TorchHelper.save_network(model, OSHelper.path_join(opt.work_space_dir, f"net_{epoch - 1}.pth"))


def __serialize_option(opt: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    message = '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        if parser is not None:
            default = parser.get_default(k)
            if v != default and default is not None:
                comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


if __name__ == '__main__':
    main()
