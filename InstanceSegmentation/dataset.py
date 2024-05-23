import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_ids, data_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, show=False):
        """
        针对语义分割的数据集类，继承自torch.utils.data.Dataset类
        参数说明：
        img_ids: 图像的ID
        data_dir: 数据集的路径
        img_ext: 图像的扩展名
        mask_ext: 掩码的扩展名
        num_classes: 类别数
        transform: 数据增强，利用albumentations库
        """
        super().__init__()
        self.img_ids = img_ids
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.show = show

    def __len__(self):
        # 展示数据中一共有多少个样本
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.data_dir, img_id + self.img_ext))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].to(dtype=torch.float32) / 255 # 将mask转换为float32类型，因为ToTensor()会将img转换为float32类型
            if self.show:
                cv2.imshow('img', img.numpy().transpose(1, 2, 0)[:, :, ::-1])
                cv2.imshow('mask', mask.numpy().transpose(1, 2, 0)[:, :, ::-1])
                if cv2.waitKey(0) or 0xFF == 'q':
                    cv2.destroyAllWindows()
        return img, mask, {'img_id': img_id} # img: [C, H, W], mask: [1, H, W]