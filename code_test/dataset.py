#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

from Dataset.TestDataset import TestDataset
from Dataset.TrainingDataset import TrainingDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    dataset = TrainingDataset(data_root="../work_space/data", preload_dataset=False, ret_dsize=224)
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

    for i, (imgs, labels) in enumerate(dataloader):
        labels = labels.permute(0, 2, 3, 1)
        imgs = imgs.permute(0, 2, 3, 1)
        imgs = imgs.numpy() * dataset.NORMALIZATION_STD + dataset.NORMALIZATION_MEAN
        B = imgs.shape[0]
        for j in range(B):
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(imgs[j])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(labels[j])
            plt.axis('off')
            plt.savefig(f"{i}_{j}.png")
            plt.close(fig)
            plt.close()
    pass


if __name__ == '__main__':
    main()