"""
=====================================
机器人视觉点动抓取
Robot：Hans Robot - Elfin
Camera：Gemini2
Version：V1.0
Copyright：XMU-GZW
=====================================
方法：
(1) 相机获取RGB和深度图像，并对齐显示
(2) 鼠标点击事件获取像素点坐标，并将其转换为机器人坐标系下的坐标
(3) 显示机器人坐标系下的坐标，机器人抓取物体

主要代码功能：
1. 机器人参数初始化 - IP和端口、工作空间、机械臂末端姿态
2. 机器人运动 - 连接、运动速率调节、回零、打开夹爪
3. 像素数据初始化 - 手动选择的像素点、坐标字符串
4. 相机坐标系下的像素点转换为机器人坐标系下的坐标
5. OpenCV窗口的回调函数 - 鼠标点击事件
6. 图像显示 - 将深度图转换为彩色并叠加到彩色图上，鼠标点击后显示机器人坐标系下的坐标

需要修改部分：
1. py文件路径
2. 机器人IP和端口
3. 工作空间
4. 机械臂末端姿态（一般不需要改）

版本文件需求：
1. Python 3.8 (强制要求，因为HansRobotSDK只支持Python3.8)
2. Numpy 1.23.5（广播计算问题，版本不同计算方法不同，换版本可能要Debug）
3. OrbbecSDK cp3.8-win-amd64 & utils.py（官方提供）
4. Gemini_camera.py（Hans_Robot.py中已集成，XMU-GZW）
5. HansRobotSDK V1.0.0.3（官方提供）
6. Hans_Robot.py（XMU-GZW）
=====================================
"""

#!/usr/bin/env python
import sys
sys.path.append('../') # 添加上级目录到系统路径中（======== 修改 1 ========）
import numpy as np
import cv2
from Hans_Robot import HansRobot as Robot

## 1. 机器人参数初始化
# 初始化参数
# 设置机器人IP和端口（======== 修改 2 ========）
tcp_host_ip = '192.168.4.40'
tcp_port = 10003
# 设置工作空间（======== 修改 3 ========）
workspace_limits = np.asarray([[500, 800], [-300, 0], [70, 200]]) # Cols: min max, Rows: x y z (机器人坐标系下的工作空间)
workspace_limits_input = [[workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][1]], [workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], [0, 0, 0, 0, 0, 0]] # 修改为Hans机器人要求的输入格式
tool_orientation = [88, 20, 88] # 机械臂末端为水平姿态（======== 修改 4 ========）
## 2. 机器人运动
# 机器人连接
print('Connecting to robot...')
robot = Robot(tcp_host_ip, tcp_port, workspace_limits=workspace_limits_input, TcpName="TCP_1", is_use_effector=False, is_use_camera=True)
robot.Connect()
# 机器人运动速率调节
Override = 0.1
# # 机器人打开夹爪
robot.OpenEndEffector()
# 机器人到达抓取起始点
robot.GoHome(Override=Override)

## 3. 像素数据初始化
click_point_pix = None # 初始化手动选择的像素点
target_position_str = [None] # 初始化用于在图像上显示的机器人坐标系下的坐标字符串

## 4. 相机坐标系下的像素点转换为机器人坐标系下的坐标
def camera2robot(click_point_pix, camera_depth_img, robot):
    click_z = camera_depth_img[click_point_pix[1]][click_point_pix[0]] * robot.cam_depth_scale # 获取深度图像中的深度值，并乘以深度图像的缩放系数
    if click_z == 0: # 如果深度值为0，说明深度图像中没有检测到点，跳过
        return [click_point_pix[0], click_point_pix[1], 0]
    click_x = np.multiply(click_point_pix[0] - robot.cam_intrinsics[0][2], click_z / robot.cam_intrinsics[0][0]) # 计算照片上的点与照片中心的水平距离，用这个距离除以焦距，然后乘以这个点到相机的实际距离，就得到了这个点到相机的实际水平距离
    click_y = np.multiply(click_point_pix[1] - robot.cam_intrinsics[1][2], click_z / robot.cam_intrinsics[1][1]) # 计算照片上的点与照片中心的垂直距离，用这个距离除以焦距，然后乘以这个点到相机的实际距离，就得到了这个点到相机的实际垂直距禭
    click_point = np.asarray([click_x, click_y, click_z]) # 组合X、Y、Z坐标
    click_point.shape = (3, 1) # 将click_point的形状变为3x1
    # 将手动选择的像素点从相机坐标系转换到机器人坐标系
    camera2robot = robot.cam_pose # 相机坐标系到机器人坐标系的变换矩阵
    # 相机坐标系到机器人坐标系的变换矩阵：camera2robot左上角3x3子矩阵为旋转矩阵R，矩阵最右边的一列（除了最后一行）为平移向量t
    # 先对手动选择的像素点相机坐标系下的坐标进行旋转变换，然后加上平移向量，得到在机器人坐标系下的坐标
    target_position = np.dot(camera2robot[0:3, 0:3], click_point) + camera2robot[0:3, 3:] # 旋转变换 + 平移向量
    target_position = target_position[0:3, 0] # 取出坐标系变换后的前三个元素，即为在机器人坐标系下的X、Y、Z坐标
    print(f"选择点的坐标为：({target_position[0]}, {target_position[1]}, {target_position[2]})")
    return target_position

## 5. OpenCV窗口的回调函数
def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # 判断鼠标左键是否按下
        global robot, click_point_pix, target_position_str # 全局变量
        click_point_pix = (x, y)
        # 获取手动选择的像素点在机器人坐标系下的坐标
        target_position = camera2robot(click_point_pix, camera_depth_img, robot)
        target_position_str = [f"Select Pixel:", f"x={target_position[0]}", f"y={target_position[1]}", f"z={target_position[2]}"] # 用于在图像上显示的机器人坐标系下的坐标字符串
        robot.Grasp_plane([target_position[0]+5,target_position[1],target_position[2]], Override=Override) # 机器人抓取后不动
        # robot.Grasp([target_position[0],target_position[1],target_position[2]], Override=Override) # 机器人抓取后将物体放在指定位置

## 6. 图像显示
cv2.namedWindow('RGB & Depth') # 创建图像窗口
cv2.setMouseCallback('RGB & Depth', mouseclick_callback) # 设置鼠标回调函数，RGB & Depth为窗口名，mouseclick_callback为回调函数

while True:
    camera_color_img, camera_depth_img = robot.get_camera_data() # 获取相机数据流，其中相机获取的RGB图像已转变为BGR格式
    # 将深度图转换为彩色并叠加到彩色图上
    # 由于深度图的值可能非常大，直接显示可能会看不到任何细节，所以需要将深度图的值归一化到0-255的范围内，并转换为8位无符号整型
    depth_colored = robot.camera.normalize_depth(camera_depth_img) # 归一化深度值
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET) # 将归一化后的深度图转换为伪彩色图像，以更好地观察深度变化
    overlay = cv2.addWeighted(camera_color_img, 0.2, depth_colored, 0.8, 0) # 将彩色图像与深度图像叠加

    # 在图像的左上角显示机器人坐标系下的坐标
    pixel_x, pixel_y = 10, 30 # 设置显示坐标的位置
    for line in target_position_str: # 遍历坐标字符串列表
        cv2.putText(overlay, line, (pixel_x, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # 0.5为字体大小，(0, 255, 0)为颜色，1为字体粗细
        pixel_y += 20 # 设置下一行坐标的位置
    cv2.imshow('RGB & Depth', overlay) # 显示图像
    if cv2.waitKey(1) == ord('c'): # 按下c键退出
        break
cv2.destroyAllWindows() # 关闭所有窗口
