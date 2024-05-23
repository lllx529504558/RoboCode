import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# from models import UNet as Net
from models import DeepLab as Net
# from models import MedT as Net
from dataset import CustomDataset
from metrics import iou_score
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument('--network', type=str, default='DeepLabV3Plus',
                        help='Network name in models')
    parser.add_argument('--backbone', type=str, default='ResNet18',
                            help='Backbone network for the model')
    parser.add_argument('--model_dir', type=str,
                        default='trained_models/DeepLabV3Plus_ResNet18_epoch_334_iou_0.9145.pth',
                        help='Path to the trained model')
    parser.add_argument('--input_w', type=int, default=256,
                        help='Input image size for the network')
    parser.add_argument('--input_h', type=int, default=256,
                        help='Input image size for the network')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes in the dataset')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='Number of channels in the input image')
    # 验证参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='dataset/cropped_image',
                        help='Directory for the dataset')
    parser.add_argument('--mask_dir', type=str, default='dataset/cropped_mask',
                        help='Directory for the mask dataset')
    parser.add_argument('--image_ext', type=str, default='.jpg',
                        help='Directory for the training dataset')
    parser.add_argument('--mask_ext', type=str, default='.png',
                        help='Directory for the validation dataset')
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the DataLoader')
    parser.add_argument('--save_dir', type=str, default='inference',
                        help='Directory for saving models')
    parser.add_argument('--deep_supervision', type=bool, default=False,
                        help='Deep supervision for the training')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # 运行设备
    if torch.cuda.is_available() and not args.force_cpu:
        print(f"CUDA detected. Running with {torch.cuda.get_device_name(0)} acceleration.")
        device = torch.device("cuda")
        cudnn.benchmark = True
    elif args.force_cpu:
        print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        print("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")

    cudnn.benchmark = True

    # 创建模型
    print("=> creating model %s" % args.network)
    if args.backbone is not None:
        model = Net.__dict__[args.network](in_channels=args.num_channels, num_classes=args.num_classes, backbone=args.backbone)
    elif args.deep_supervision:
        model = Net.__dict__[args.network](in_channels=args.num_channels, num_classes=args.num_classes, deep_supervision=args.deep_supervision)
    else:
        model = Net.__dict__[args.network](in_channels=args.num_channels, num_classes=args.num_classes)
    model = model.to(device)

    # 加载数据集
    img_ids = glob(os.path.join(args.data_dir, '*' + args.image_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    model.load_state_dict(torch.load(args.model_dir))
    model.eval()

    val_transform = A.Compose([
        A.Resize(256, 256),
        # A.CenterCrop(args.input_h, args.input_w),
        # A.MaskDropout(max_objects=1, image_fill_value='inpaint', p=1),
        # A.Normalize(mean=(0.4758, 0.4552, 0.4241), std=(0.3141, 0.2958, 0.2729)), # 100张
        A.Normalize(mean=(0.4105, 0.3815, 0.3482), std=(0.3539, 0.3232, 0.2745)), # 40张
        ToTensorV2(transpose_mask=True),
    ])

    val_dataset = CustomDataset(
        img_ids = img_ids,
        data_dir = args.data_dir,
        mask_dir = args.mask_dir,
        img_ext = args.image_ext,
        mask_ext = args.mask_ext,
        num_classes = args.num_classes,
        transform = val_transform,
        # show = True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    avg_meter = AverageMeter()
    # 创建保存目录
    if args.backbone is not None:
        for c in range(args.num_classes):
            os.makedirs(os.path.join(args.save_dir, args.network, args.backbone, str(c)), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, args.backbone, 'segmentation'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, args.backbone, 'overlayed'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, args.backbone, 'augmentation'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, args.backbone, 'mask'), exist_ok=True)
    else:
        for c in range(args.num_classes):
            os.makedirs(os.path.join(args.save_dir, args.network), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, 'segmentation'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, 'overlayed'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, 'augmentation'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, args.network, 'mask'), exist_ok=True)
    # 为每个实例生成一个随机颜色
    colors = np.random.randint(0, 255, size=(args.num_classes, 3))
    # 验证
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            if args.deep_supervision:
                output = torch.squeeze(model(input)[-1])
            else:
                output = torch.squeeze(model(input))
                
            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            # 初始化空白彩色图像用于绘制分割结果
            segmentation_result = np.zeros_like(input.cpu().numpy())

            for i in range(len(output)):
                input_s = input.cpu().numpy()[i]
                norm_input_s = (input_s - input_s.min()) / (input_s.max() - input_s.min()) * 255 # 转换到0-255之间
                output_s = output[i]
                segmentation_result_s = segmentation_result[i]
                for c in range(args.num_classes): # 遍历每个实例
                    if args.num_classes == 1: # 二分类
                        mask = output_s  # 获取当前实例的分割掩码
                    else: # 多分类
                        mask = output_s[c]  # 获取当前实例的分割掩码
                    if args.backbone is not None:
                        cv2.imwrite(os.path.join(args.save_dir, args.network, args.backbone, str(c), meta['img_id'][i] + '.jpg'),
                            (mask * 255).astype('uint8'))
                    else:
                        cv2.imwrite(os.path.join(args.save_dir, args.network, str(c), meta['img_id'][i] + '.jpg'),
                            (mask * 255).astype('uint8'))
                    
                    color = colors[c]  # 为当前实例分配颜色
                    # 将当前实例的分割区域着色
                    for j in range(3):
                        segmentation_result_s[j, mask > 0.9] = color[j] # 为mask>0的区域着色
                # 保存分割结果
                if args.backbone is not None:
                    cv2.imwrite(os.path.join(args.save_dir, args.network, args.backbone, 'segmentation', meta['img_id'][i] + '.jpg'), cv2.cvtColor(segmentation_result_s.astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(os.path.join(args.save_dir, args.network, 'segmentation', meta['img_id'][i] + '.jpg'), cv2.cvtColor(segmentation_result_s.astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                # 将着色的分割结果与原始图像叠加（可选）
                alpha = 0.5  # 设置叠加的透明度
                overlayed_result = cv2.addWeighted(norm_input_s, alpha, segmentation_result_s, 1 - alpha, 0)
                if args.backbone is not None:
                    cv2.imwrite(os.path.join(args.save_dir, args.network, args.backbone, 'overlayed', meta['img_id'][i] + '.jpg'), cv2.cvtColor(overlayed_result.astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                    # 保存增强后的图像
                    cv2.imwrite(os.path.join(args.save_dir, args.network, args.backbone, 'augmentation', meta['img_id'][i] + '.jpg'), cv2.cvtColor((norm_input_s).astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)) # 保存增强后的图像
                    cv2.imwrite(os.path.join(args.save_dir, args.network, args.backbone, 'mask', meta['img_id'][i] + '_mask.jpg'), cv2.cvtColor((target[i].cpu().numpy() * 255).astype('uint8'), cv2.COLOR_RGB2BGR)) # 保存增强后的mask
                else:
                    cv2.imwrite(os.path.join(args.save_dir, args.network, 'overlayed', meta['img_id'][i] + '.jpg'), cv2.cvtColor(overlayed_result.astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                    # 保存增强后的图像
                    cv2.imwrite(os.path.join(args.save_dir, args.network, 'augmentation', meta['img_id'][i] + '.jpg'), cv2.cvtColor((norm_input_s).astype('uint8').transpose(1, 2, 0), cv2.COLOR_RGB2BGR)) # 保存增强后的图像
                    cv2.imwrite(os.path.join(args.save_dir, args.network, 'mask', meta['img_id'][i] + '_mask.jpg'), cv2.cvtColor((target[i].cpu().numpy() * 255).astype('uint8'), cv2.COLOR_RGB2BGR))
    print('IoU: %.4f' % avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()