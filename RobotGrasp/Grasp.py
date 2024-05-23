"""
=====================================
Hans Robot 封装函数库
Robot：Hans Robot - Elfin
Camera：Gemini2
Version：V1.0
Copyright：XMU-GZW
=====================================

主要代码功能：
1. 相机坐标系到机器人坐标系的转换
2. 绘制抓取点
3. 计算质心

工作原理：
连接机械臂和相机，通过相机拍摄物体，利用DeepLabV3Plus模型检测目标物体，计算物体的质心，然后将质心转换到机器人坐标系下，机器人移动到质心位置进行抓取。
"""

import cv2
import torch
import numpy as np
import time
from Hans_Robot import HansRobot as Robot
from inference.point_generator import PointGenerator


def camera2robot(click_point_pix, camera_depth_img, pointGen):
    click_z = camera_depth_img[click_point_pix[1]][click_point_pix[0]] * robot.cam_depth_scale # 获取深度图像中的深度值，并乘以深度图像的缩放系数
    if click_z == 0: # 如果深度值为0，说明深度图像中没有检测到点，跳过
        return [click_point_pix[0], click_point_pix[1], 0]
    click_x = np.multiply(click_point_pix[0] - pointGen.cam_intrinsics[0][2], click_z / pointGen.cam_intrinsics[0][0]) # 计算照片上的点与照片中心的水平距离，用这个距离除以焦距，然后乘以这个点到相机的实际距离，就得到了这个点到相机的实际水平距离
    click_y = np.multiply(click_point_pix[1] - pointGen.cam_intrinsics[1][2], click_z / pointGen.cam_intrinsics[1][1]) # 计算照片上的点与照片中心的垂直距离，用这个距离除以焦距，然后乘以这个点到相机的实际距离，就得到了这个点到相机的实际垂直距禭
    click_point = np.asarray([click_x, click_y, click_z]) # 组合X、Y、Z坐标
    click_point.shape = (3, 1) # 将click_point的形状变为3x1
    # 将手动选择的像素点从相机坐标系转换到机器人坐标系
    camera2robot = pointGen.cam_pose # 相机坐标系到机器人坐标系的变换矩阵
    # 相机坐标系到机器人坐标系的变换矩阵：camera2robot左上角3x3子矩阵为旋转矩阵R，矩阵最右边的一列（除了最后一行）为平移向量t
    # 先对手动选择的像素点相机坐标系下的坐标进行旋转变换，然后加上平移向量，得到在机器人坐标系下的坐标
    target_position = np.dot(camera2robot[0:3, 0:3], click_point) + camera2robot[0:3, 3:] # 旋转变换 + 平移向量
    target_position = target_position[0:3, 0] # 取出坐标系变换后的前三个元素，即为在机器人坐标系下的X、Y、Z坐标
    print(f"目标点的坐标为：({target_position[0]}, {target_position[1]}, {target_position[2]})")
    return target_position

def draw_point(rgb, point_list, idx, color=(0, 255, 0), radius=3, thickness=-1, wait=False):
    # 绘制每个点
    for i, (x, y) in enumerate(point_list):
        if i == idx:
            cv2.circle(rgb, (int(x), int(y)), radius, (0, 0, 255), thickness) # 画圆来标记点，颜色设置为红色，圆的半径为3，线宽为-1表示填充圆
        else:
            cv2.circle(rgb, (int(x), int(y)), radius, color, thickness) # 画圆来标记点，颜色设置为绿色，圆的半径为3，线宽为-1表示填充圆
        # 添加文本标签，颜色设置为白色
        cv2.putText(rgb, str(i + 1), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # 显示图像
    cv2.imshow('Points on Image', rgb)
    if wait:
        if cv2.waitKey(0) & 0xFF == ord('c'):
            cv2.destroyAllWindows()
    else:
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

def calculate_centroids(pointGen):
    # 计算抓取点
    rgb, depth, rgb_crop, output = pointGen.generate()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = torch.sigmoid(output).cpu().numpy().squeeze()
    mask = (mask > 0.9).astype(np.uint8)
    # 标记每个区域
    labeled_array, num_features = pointGen.label_region(mask)
    print(f"Found {num_features} features")
    # 计算每个区域的质心
    centroids_list = pointGen.point(mask, labeled_array, num_features)
    centroids_list = sorted(centroids_list, key=lambda item: (item[1], item[0]))
    return rgb, depth, centroids_list



# 主函数
# --------------- 初始化设置 ---------------
tcp_host_ip = '192.168.4.40' # IP and port to robot arm as TCP client
tcp_port = 10003
tool_orientation = [88, 20, 88]
workspace_limits = np.asarray([[400, 700], [-300, 50], [70, 200]]) # Cols: min max, Rows: x y z (机器人坐标系下的工作空间)
workspace_limits_input = [[workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][1]], [workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], [0, 0, 0, 0, 0, 0]] # 修改为Hans机器人要求的输入格式
# ---------------------------------------------

# 机器人移动到初始位置
print('Connecting to robot...')
robot = Robot(tcp_host_ip, tcp_port, workspace_limits=workspace_limits_input, TcpName="TCP_1", is_use_effector=False, is_use_camera=False)
robot.Connect()
Override = 0.1 # 机器人移动速度
robot.OpenEndEffector()
robot.GoHome(Override=Override) # 机器人回到初始位置

# 读取模型
model_path = r'\RoboGrasp\InstanceSegmentation\models\trained_models\DeepLabV3Plus_ResNet18_epoch_334_iou_0.9145.pth'
pointGen = PointGenerator(model_path)
# 计算抓取点
rgb, depth, centroids_list = calculate_centroids(pointGen)

if len(centroids_list) == 7:
    for i in range(len(centroids_list))[::-1]:
        # 显示图像
        draw_point(rgb, centroids_list, i, wait=False)
        # 获取像素点坐标
        point_pix = (int(centroids_list[i][0]), int(centroids_list[i][1]))
        target_position = camera2robot(point_pix, depth, pointGen)
        # robot.Grasp_plane([target_position[0], target_position[1], target_position[2]], Override=Override)
        robot.Grasp([target_position[0], target_position[1], target_position[2]-5], Override=Override)
        print(f"Grasp point {i + 1}: ({target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]-5:.2f})")
        # break
        robot.OpenEndEffector()
        robot.GoHome(Override=Override*2)
        while not robot.IsMotionDone():
            pass
        # 计算新的抓取点
        rgb, depth, centroids_list = calculate_centroids(pointGen)
else:
    print(f"仅检测到{len(centroids_list)}个抓取点")
    draw_point(rgb, centroids_list, None, wait=True)

time.sleep(1)
robot.GoHome(Override=Override)

