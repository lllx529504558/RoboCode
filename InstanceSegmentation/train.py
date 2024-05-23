import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from dataset import CustomDataset
from collections import OrderedDict
from tqdm import tqdm
from utils import AverageMeter
from metrics import iou_score
import losses
from models import DeepLab as Net
# from models import UNet as Net
from sklearn.model_selection import train_test_split
from glob import glob
import os
from torch.optim import lr_scheduler
import pandas as pd
import time

# 设置网络参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # 模型参数
    parser.add_argument('--network', type=str,
                        # default='UNetPlus',
                        # default='NestedUNet',
                        # default='UNet',
                        default='DeepLabV3Plus',
                        help='Network name in models')
    parser.add_argument('--backbone', type=str,
                        default='ResNet18',
                        default=None,
                        help='Backbone network for the model')
    parser.add_argument('--input_w', type=int, default=256,
                        help='Input image size for the network')
    parser.add_argument('--input_h', type=int, default=256,
                        help='Input image size for the network')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of classes in the dataset')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='Number of channels in the input image')
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='dataset/cropped_image',
                        help='Directory for the dataset')
    parser.add_argument('--mask_dir', type=str, default='dataset/cropped_mask',
                        help='Directory for the mask dataset')
    parser.add_argument('--image_ext', type=str, default='.jpg',
                        help='Directory for the training dataset')
    parser.add_argument('-mask_ext', type=str, default='.png',
                        help='Directory for the validation dataset')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Training epochs')
    # 损失函数参数
    parser.add_argument('--loss', type=str, default='BCEDice',
                        help='Loss function for the training. (CrossEntropy, BCEDice, MultiBce, LovaszHinge)')
    # 优化器参数
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for the training')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for the training')
    # 学习率调度器参数
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='Learning rate scheduler for the training')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for the learning rate scheduler')
    parser.add_argument('--early_stopping', type=int, default=50,
                        help='Early stopping for the training')
    # 其他参数
    parser.add_argument('--seed', type=int, default=1412,
                        help='Random seed for the training')
    parser.add_argument('--save_dir', type=str, default='trained_models',
                        help='Directory for saving models')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the DataLoader')
    parser.add_argument('--deep_supervision', type=bool, default=False,
                        help='Deep supervision for the training')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    # 返回参数
    args = parser.parse_args()
    return args

def train(args, train_loader, model, criterion, optimizer, device):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)
        # 计算输出
        if args.deep_supervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                output = torch.squeeze(output) # [N, 1, H, W] -> [N, H, W]
                if output.size(0) != target.size(0):
                    target = target.squeeze(0)
                loss += criterion(output, target) # target: [N, H, W]
            loss /= len(outputs)
            iou = iou_score(torch.squeeze(outputs[-1]), target)
        else:
            output = torch.squeeze(model(input)) # [N, 1, H, W] -> [N, H, W]
            if output.size(0) != target.size(0):
                target = target.squeeze(0)
            loss = criterion(output, target) # target: [N, H, W]
            iou = iou_score(output, target)
        
        # 计算梯度，迭代最优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(args, val_loader, model, criterion, device):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # 计算输出
            if args.deep_supervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    output = torch.squeeze(output) # [N, 1, H, W] -> [N, H, W]
                    if output.size(0) != target.size(0):
                        target = target.squeeze(0)
                    loss += criterion(output, target) # target: [N, H, W]
                loss /= len(outputs)
                iou = iou_score(torch.squeeze(outputs[-1]), target)
            else:
                output = torch.squeeze(model(input)) # [N, 1, H, W] -> [N, H, W]
                if output.size(0) != target.size(0):
                    target = target.squeeze(0)
                loss = criterion(output, target) # target: [N, H, W]
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    # 记录当前时间
    time_str = time.strftime("%Y%m%d_%H%M", time.localtime())
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
    # 创建模型
    print("=> creating model %s" % args.network)
    if args.backbone is not None:
        model = Net.__dict__[args.network](in_channels=args.num_channels, num_classes=args.num_classes, backbone=args.backbone)
    elif args.deep_supervision:
        model = Net.__dict__[args.network](in_channels=args.num_channels, num_classes=args.num_classes, deep_supervision=args.deep_supervision)
    else:
        model = Net.__dict__[args.network](in_channels=args.num_channels, num_classes=args.num_classes)
    model = model.to(device)
    # 损失函数
    criterion = losses.__dict__[args.loss]().cuda()
    # 优化器
    params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))
    # 学习率调度器
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                   verbose=1, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    
    # 加载数据集
    img_ids = glob(os.path.join(args.data_dir, '*' + args.image_ext))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=args.seed)
    
    # 数据增强
    train_transform = A.Compose([
        A.Rotate(limit=90, p=0.5), # 随机旋转，-90~90度
        A.RandomResizedCrop(width=args.input_w, height=args.input_h, scale=(0.3, 1), ratio=(0.7, 1.3)), # 随机裁剪0.6~1倍原图尺寸的区域，纵横比为0.9~1.1, 然后resize到指定尺寸，
        # A.RandomResizedCrop(size=(args.input_w, args.input_h), scale=(0.3, 1), ratio=(0.7, 1.3)), # 随机裁剪0.6~1倍原图尺寸的区域，纵横比为0.9~1.1, 然后resize到指定尺寸，
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5), # 随机移动RGB通道，范围-25~25
        A.HorizontalFlip(p=0.5), # 水平翻转
        A.RandomBrightnessContrast(p=0.2), # 随机亮度对比度
        # A.MaskDropout(max_objects=3, image_fill_value='inpaint', p=0.5), # 随机去掉图像的标签区域，然后用inpaint填充，效果不太好哦（inpaint效果就不好）
        A.Normalize(mean=(0.4105, 0.3815, 0.3482), std=(0.3539, 0.3232, 0.2745)), # 40张
        # A.Normalize(mean=(0.4758, 0.4552, 0.4241), std=(0.3141, 0.2958, 0.2729)), # 100张
        ToTensorV2(transpose_mask=True),
    ])
    val_transform = A.Compose([
        # A.Resize(args.input_h, args.input_w),
        A.CenterCrop(args.input_h, args.input_w),
        A.Normalize(mean=(0.4105, 0.3815, 0.3482), std=(0.3539, 0.3232, 0.2745)), # 40张
        # A.Normalize(mean=(0.4758, 0.4552, 0.4241), std=(0.3141, 0.2958, 0.2729)), # 100张
        ToTensorV2(transpose_mask=True),
    ])
    # 读取数据集
    train_datasets = CustomDataset(
        img_ids = train_img_ids,
        data_dir = args.data_dir,
        mask_dir = args.mask_dir,
        img_ext = args.image_ext,
        mask_ext = args.mask_ext,
        num_classes = args.num_classes,
        transform = train_transform
    )
    val_datasets = CustomDataset(
        img_ids = val_img_ids,
        data_dir = args.data_dir,
        mask_dir = args.mask_dir,
        img_ext = args.image_ext,
        mask_ext = args.mask_ext,
        num_classes = args.num_classes,
        transform = val_transform
    )
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    # 初始化日志
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('train_loss', []),
        ('train_iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])
    best_iou = 0
    trigger = 0
    # 模型训练
    for epoch in range(args.epochs):
        print(f'Epoch [{epoch+1}/{args.epochs}]')
        # 训练一步
        train_log = train(args, train_loader, model, criterion, optimizer, device)
        # 验证一步
        val_log = validate(args, val_loader, model, criterion, device)
        if args.scheduler.lower() == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler.lower() == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        # 记录日志
        log['epoch'].append(epoch+1)
        log['lr'].append(args.lr)
        log['train_loss'].append(train_log['loss'])
        log['train_iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        # early stopping计数器
        trigger += 1
        # 保存最好的模型
        if val_log['iou'] > best_iou:
            best_iou = val_log['iou']
            if args.backbone is not None:
                torch.save(model.state_dict(), f'{args.save_dir}/{args.network}_{args.backbone}_epoch_{epoch+1}_iou_{best_iou:.4f}.pth')
            else:
                torch.save(model.state_dict(), f'{args.save_dir}/{args.network}_epoch_{epoch+1}_iou_{best_iou:.4f}.pth')
            print("=> saved best model")
            trigger = 0
        elif (epoch+1) % 10 == 0:
            current_iou = val_log['iou']
            if args.backbone is not None:
                torch.save(model.state_dict(), f'{args.save_dir}/{args.network}_{args.backbone}_epoch_{epoch+1}_iou_{current_iou:.4f}_autosave.pth')
            else:
                torch.save(model.state_dict(), f'{args.save_dir}/{args.network}_epoch_{epoch+1}_iou_{current_iou:.4f}_autosave.pth')
            print(f"=> saved model, {epoch+1} epoch")
        # early stopping
        if args.early_stopping >= 0 and trigger >= args.early_stopping:
            # 保存训练过程
            if args.backbone is not None:
                pd.DataFrame(log).to_csv(f'{args.save_dir}/log_{args.network}_{args.backbone}_{str(time_str)}.csv', index=False)
            else:
                pd.DataFrame(log).to_csv(f'{args.save_dir}/log_{args.network}_{str(time_str)}.csv', index=False)
            print("=> early stopping")
            break
        # 释放内存
        torch.cuda.empty_cache()
    # 保存训练过程
    if args.backbone is not None:
        pd.DataFrame(log).to_csv(f'{args.save_dir}/log_{args.network}_{args.backbone}_{str(time_str)}.csv', index=False)
    else:
        pd.DataFrame(log).to_csv(f'{args.save_dir}/log_{args.network}_{str(time_str)}.csv', index=False)


if __name__ == '__main__':
    main()