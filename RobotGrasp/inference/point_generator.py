import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label, center_of_mass

# from hardware.Gemini_camera import OrbbecGeminiCamera as Camera
from hardware.ZED2i_camera import ZED2iCamera as Camera
from hardware.device import get_device
from utils.data.camera_data import CameraData
from inference.models import DeepLab as Net

class PointGenerator():
    def __init__(self, model_path, device_id='34680631', image_resolution='HD1080', image_fps=5, net='DeepLabV3Plus', backbone='ResNet18',in_channels=3, num_classes=1, crop_size=400, reszie=256, force_cpu=False):
        # self.camera = Camera(device_id=cam_id) # OrbbecGeminiCamera
        self.camera = Camera(device_id=device_id,
                             image_resolution=image_resolution,
                             image_fps=image_fps) # ZED2iCamera
        self.camera.connect(enable_fill_mode=True, depth_mode='NEURAL+', depth_unit='mm', depth_min=300, depth_max=1500) # ZED2i相机连接
        self.cam_intrinsics = self.camera.get_intrinsics() # 获取相机内参，封装为特定格式
        self.cam_data = CameraData(include_depth=True, include_rgb=True)
        self.model_path = model_path
        self.device = get_device(force_cpu)
        self.model = Net.__dict__[net](in_channels=in_channels, num_classes=num_classes, backbone=backbone)
        self.crop_size = crop_size
        self.reszie = reszie

        # Load camera pose and depth scale (from running calibration)
        self.cam_pose = np.loadtxt('E:\workspace\Anaconda\Robot\GRCNN\Calibrate\camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('E:\workspace\Anaconda\Robot\GRCNN\Calibrate\camera_depth_scale.txt', delimiter=' ')

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = image_bundle['aligned_depth']
        # 设置图像预处理
        transform = A.Compose([
                                A.CenterCrop(self.crop_size, self.crop_size),
                                A.Resize(self.reszie, self.reszie),
                                # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # RGB
                                # A.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]), # BGR
                                A.Normalize(mean=(0.4105, 0.3815, 0.3482), std=(0.3539, 0.3232, 0.2745)),
                                ToTensorV2(),
                             ])
        # 应用转换
        input_tensor = transform(image=rgb)['image']
        input_tensor = input_tensor.unsqueeze(0)  # 增加batch维度
        input_tensor = input_tensor.to(self.device)
        # 加载模型
        print('Loading model... ')
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        # 确保模型处于评估模式
        self.model.eval()
        # 执行预测
        with torch.no_grad():
            output = self.model(input_tensor)
        return rgb, depth, input_tensor.cpu().numpy(), output
    
    def get_original_coordinates(self, x_prime, y_prime, original_width, original_height):
        # 将缩放后的坐标转换回320x320尺寸的坐标
        x_crop = x_prime * (self.crop_size / self.reszie)
        y_crop = y_prime * (self.crop_size / self.reszie)

        # 计算原始图像中的坐标
        x_original = x_crop + (original_width - self.crop_size) / 2
        y_original = y_crop + (original_height - self.crop_size) / 2

        return x_original, y_original
    
    # 标记每个区域
    def label_region(self, mask):
        labeled_array, num_features = label(mask)
        return labeled_array, num_features
    
    def point(self, mask, labeled_array, num_features, show_output=False):
        # 计算每个区域的质心
        centroids = center_of_mass(mask, labeled_array, range(1, num_features + 1))
        centroids_list = []
        # 打印每个区域的质心坐标
        for i, centroid in enumerate(centroids):
            x_original, y_original = self.get_original_coordinates(centroid[1], centroid[0], 1920, 1080)
            centroids_list.append((x_original, y_original))
            if show_output:
                print(f"Region {i + 1}: Centroid at (x, y) = ({x_original:.2f}, {y_original:.2f})")
        return centroids_list
    
    # 为每个实例生成一个随机颜色
    def generate_colors(self, mask, labeled_array, num_features):
        # 为每个实例生成一个随机颜色
        colors = np.random.randint(0, 255, size=(num_features + 1, 3))
        segmentation_result = np.zeros(mask.shape + (3,), dtype=np.uint8)
        for label_num in range(1, num_features + 1):
            mask = labeled_array == label_num
            segmentation_result[mask] = colors[label_num]
        segmentation_result = segmentation_result.astype(np.uint8)
        return segmentation_result

if __name__ == '__main__':
    model_path = r'E:\workspace\Anaconda\Robot\InstanceSegmentation\models\trained_models\DeepLabV3Plus_ResNet18_epoch_334_iou_0.9145.pth'
    pointGen = PointGenerator(model_path)
    rgb, depth, rgb_crop, output = pointGen.generate()
    rgb_crop = cv2.cvtColor(rgb_crop.squeeze().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    mask = torch.sigmoid(output).cpu().numpy().squeeze()
    mask = (mask > 0.9).astype(np.uint8)
    # 标记每个区域
    labeled_array, num_features = pointGen.label_region(mask)
    print(f"Found {num_features} features")
    # 计算每个区域的质心
    centroids_list = pointGen.point(mask, labeled_array, num_features)
    segmentation_result = pointGen.generate_colors(mask, labeled_array, num_features)
    # 显示图像
    cv2.imshow('RGB Image', rgb_crop)
    cv2.imshow('Mask Image', segmentation_result)
    cv2.waitKey(0)  # 等待任何键盘输入
    cv2.destroyAllWindows()  # 之后关闭显示窗口

