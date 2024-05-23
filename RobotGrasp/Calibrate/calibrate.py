"""
=====================================
机器人手眼标定函数
Robot：Hans Robot - Elfin
Camera：Gemini2
Version：V1.0
Copyright：XMU-GZW
=====================================
坐标系介绍：
1. 像素坐标系 - 以图像左上角原点，水平向右为X轴正方向，垂直向下为Y轴正方向
2. 图像坐标系 - 以图像中心为原点，水平向右为X轴正方向，垂直向下为Y轴正方向
3. 相机坐标系 - 物体相对于相机中心的实际坐标，水平向右为X轴正方向，垂直向下为Y轴正方向，Z轴由相机光轴位置垂直向外为正方向
4. 机器人坐标系 - 物体相对于机器人基座的实际坐标，Z轴由机器人基座向上为正方向，X和Y轴以机器人型号和安装位置为准（当前机器人采用右手坐标系，X轴正方向指向西侧）

方法：
(1) 机器人实现标定板中心点的运动
for i in range(标定点个数):
    (2) 计算标定板中心点在像素坐标系中的坐标
    (3) 计算利用相机内参矫正后的相机坐标系的坐标
    (4) 记录标定板中心点在机器人坐标系下的实际坐标值
(5) 根据多个标定板中心点的坐标值，计算 机器人坐标系->相机坐标系 的转换矩阵
(6) 利用Nelder-Mead最优化方法，优化Z轴缩放系数，使得多个标定板中心点的坐标值由机器人坐标系转换为相机坐标系后变换误差最小
(7) 获得相机Z轴缩放系数和相机姿态（相机坐标系到机器人坐标系的变换矩阵，求逆得到）

主要代码功能：
0. 先将标定板放在机械臂末端，量取标定板偏移距离，然后将标定板中心点放在相机视野中，然后运行该代码
1. 机器人参数初始化 - IP和端口、工作空间、标定间隔、标定板偏移、机械臂末端姿态、起始点位置、标定工作空间
2. 机器人手眼标定：
    2.1 机器运动初始化：机器人连接、机器人运动速率调节、机器人到达标定起始点
    2.2 机器人运动：机器人移动到标定点、找到标定板中心、机器人回到标定起始点
        PS：找到标点板的角点后，会绘制角点图像
    2.3 获取标定所需数据：
        2.3.1 利用相机获取标定板中心点在照片中的像素值（即，像素坐标系下的坐标）
        2.3.2 相机获取的标定板中心点经相机内参矫正后的像素值（即，相机坐标系下，利用相机内参矫正后的坐标）
        2.3.3 标定板中心点在机器人坐标系下的实际坐标值（已包括偏移量）
    2.4 计算实际坐标系到相机坐标系的变换误差：
        2.4.1 利用SVD计算实际坐标系到相机坐标系的刚体变换（旋转矩阵和平移向量）
        2.4.2 计算刚体变换误差（将实际坐标转换为相机坐标，并计算与相机内参矫正后坐标的RMSE）
    2.5 最优化Z轴缩放：利用Nelder-Mead最优化方法，优化Z轴缩放系数，使得刚体变换误差最小
    2.6 获得相机Z轴缩放系数和相机姿态（相机坐标系到实际坐标系的变换矩阵，求逆得到）

需要修改部分：
1. py文件路径
2. 机器人IP和端口
3. 工作空间
4. 标定间隔
5. 标定板偏移
6. 机械臂末端姿态（一般不需要改）
7. 标定内角点数
8. 如果相机获得的图像是RGB格式，需要转换为BGR格式（Gemini2获取的RGB格式已在相机封装函数中转变为BGR）
9. 保存文件路径

版本文件需求：
1. Python 3.8 (强制要求，因为HansRobotSDK只支持Python3.8)
2. Numpy 1.23.5（广播计算问题，版本不同计算方法不同，换版本可能要Debug）
3. OrbbecSDK cp3.8-win-amd64 & utils.py（官方提供）
4. Gemini_camera.py（Hans_Robot.py中已集成，XMU-GZW）
5. HansRobotSDK V1.0.0.3（官方提供）
6. Hans_Robot.py（XMU-GZW）
=====================================
"""

import sys
sys.path.append('../') # 添加上级目录到系统路径中（======== 修改 1 ========）
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from Hans_Robot import HansRobot as Robot
from scipy import optimize

## 1. 初始化参数
# 设置机器人IP和端口（======== 修改 2 ========）
tcp_host_ip = '192.168.4.40'
tcp_port = 10003
# 设置工作空间（======== 修改 3 ========）
workspace_limits = np.asarray([[500, 700], [-250, 50], [80, 180]]) # Cols: min max, Rows: x y z (机器人坐标系下的工作空间)
workspace_limits_input = [[workspace_limits[0][1], workspace_limits[1][1], workspace_limits[2][1]], [workspace_limits[0][0], workspace_limits[1][0], workspace_limits[2][0]], [0, 0, 0, 0, 0, 0]] # 修改为Hans机器人要求的输入格式
# 设置标定间隔（======== 修改 4 ========）
calib_grid_step = 100 # 标定间隔为100mm
checkerboard_offset_from_tool = [82, 0, 0] # 机械臂末端加装标定板之后，由于抓握姿态变化，标定板中心相对于夹爪末端在X轴正方向偏移了57.5 mm（======== 修改 5 ========）
tool_orientation = [88, 20, 88] # 机械臂末端为水平姿态（======== 修改 6 ========）
# 起始点位置
calib_home_ponit = [np.mean(workspace_limits[0]), np.mean(workspace_limits[1]), np.mean(workspace_limits[2])] + tool_orientation
# 标定工作空间
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], int(1 + (workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], int(1 + (workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], int(1 + (workspace_limits[2][1] - workspace_limits[2][0]) / calib_grid_step))
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
print(f"Total {num_calib_grid_pts} calibration points.")
# 生成标定点坐标
calib_grid_x.shape = (num_calib_grid_pts, 1)
calib_grid_y.shape = (num_calib_grid_pts, 1)
calib_grid_z.shape = (num_calib_grid_pts, 1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)


## 2. 机器人手眼标定开始
## 2.1 机器运动初始化
# 初始化标定数据
observed_pix = [] # 相机获取的标定板中心点的像素坐标系下的坐标值
observed_pts = [] # 相机获取的标定板中心点经相机内参矫正后的相机坐标系下的坐标值
measured_pts = [] # 标定板中心点在机器人坐标系下的实际坐标值，包括相对于机械臂末端的实际偏移量
# 机器人连接
print('Connecting to robot...')
robot = Robot(tcp_host_ip, tcp_port, workspace_limits=workspace_limits_input, TcpName="TCP_1", is_use_effector=False, is_use_camera=True)
robot.Connect()
# 机器人运动速率调节
Override = 0.1
## 2.2 机器人运动
# 机器人到达标定起始点
robot.MoveL(calib_home_ponit, Override=Override)
while not robot.IsMotionDone():
    pass
time.sleep(2) # 等待2s，待机械臂稳定
# 机器人移动到标定点
print('Collecting data...')
for calib_pt_idx in range(num_calib_grid_pts):
    tool_position = calib_grid_pts[calib_pt_idx, :]
    tool_config = [tool_position[0], tool_position[1], tool_position[2], *tool_orientation]
    print(f"Running to the {calib_pt_idx+1}th point: {tool_config} ...")
    robot.MoveL(tool_config, Override=Override)
    while not robot.IsMotionDone():
        pass
    time.sleep(3) # 等待3s，待相机观测稳定
    
    # 找到标定板中心
    checkerboard_size = (5, 5) # 设置标定内角点数，格子数减一（======== 修改 7 ========）
    # 设置角点精细化迭代终止条件：cv2.TERM_CRITERIA_EPS 表示当算法的迭代改进小于某个阈值时停止，cv2.TERM_CRITERIA_MAX_ITER 表示最大迭代次数，cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER 表示两者之一满足即停止
    # 30 表示最大迭代次数，0.001 表示算法收敛的精度，即迭代大于30次或改进小于0.001时停止
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera_color_img, camera_depth_img = robot.get_camera_data() # 获取相机数据流（RGB和Depth图像）
    camera_color_img = np.copy(camera_color_img) # 复制一份，以免内部数据发生变化，导致后续计算不兼容
    depth_image = np.clip(camera_depth_img / 1500 * 255, 0, 255).astype(np.uint8) # 深度图像归一化
    depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET) # 将归一化后的深度图转换为伪彩色图像，以更好地观察深度变化
    depth_image = cv2.addWeighted(camera_color_img, 0.5, depth_image, 0.5, 0) # 重合彩色图像与深度图像
    gray_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2GRAY) # 转为灰度图，以减少计算复杂度
    checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH) # 寻找棋盘格角点
    print(f"Find checkerboard: {checkerboard_found}.")
    ## 2.3 获取标定所需数据
    if checkerboard_found:
        # 通过迭代过程进一步精细化角点的位置，以提高标定精度
        # 输入参数：灰度图，角点，搜索窗口大小，死区，迭代终止条件
        # (5, 5) 表示搜索窗口像素大小，(-1,-1) 表示没有死区
        corners_refined = cv2.cornerSubPix(gray_data, corners, (5, 5), (-1, -1), refine_criteria)
        # 获取观测到的标定板中心点在相机坐标系下的3D坐标
        center_corner_idx = int((checkerboard_size[0] * checkerboard_size[1] - 1) / 2) # 获取标定板中心点的索引，如第13个点：5*5/2-1=12（======== 修改 ========）
        # center_corner_idx = int((checkerboard_size[0] * checkerboard_size[1]) / 2 - 4) # 获取标定板中心点的索引，如第29个点：8*8/2-4=28（======== 修改 ========）
        checkerboard_pix = np.round(corners_refined[center_corner_idx, 0, :]).astype(int) # 获取标定板中心点的像素坐标
        checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]][0] # 获取标定板中心点的深度值
        if checkerboard_z == 0: # 如果深度值为0，说明深度图像中没有检测到标定板中心点，跳过
            continue
        checkerboard_x = np.multiply(checkerboard_pix[0] - robot.cam_intrinsics[0][2], checkerboard_z / robot.cam_intrinsics[0][0]) # 计算照片上的点与照片中心的水平距离，用这个距离除以焦距，然后乘以这个点到相机的实际距离，就得到了这个点到相机的实际水平距离
        checkerboard_y = np.multiply(checkerboard_pix[1] - robot.cam_intrinsics[1][2], checkerboard_z / robot.cam_intrinsics[1][1]) # 计算照片上的点与照片中心的垂直距离，用这个距离除以焦距，然后乘以这个点到相机的实际距离，就得到了这个点到相机的实际垂直距离
        # 保存标定点和观测到的标定板中心
        observed_pix.append(checkerboard_pix) # 像素坐标系下的坐标值
        observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z]) # 相机坐标系下的坐标值
        tool_position = tool_position + checkerboard_offset_from_tool # 标定板中心点在机器人坐标系下的实际坐标值
        measured_pts.append(tool_position)
        print(f"Observed pixel: {checkerboard_pix}, observed point: {observed_pts[-1]}, measured point: {measured_pts[-1]}.")

        # 显示检测到的标定板中心点
        # Gemini2相机的封装函数已将RGB转为BGR格式，如换其他相机需转换格式（======== 修改 8 ========）
        # vis = cv2.drawChessboardCorners(camera_color_img, checkerboard_size, corners_refined, checkerboard_found) # 在图像上绘制全部角点
        # vis = cv2.drawChessboardCorners(camera_color_img, (1,1), corners_refined[center_corner_idx, :, :], checkerboard_found) # 在图像上绘制中心点
        vis = cv2.drawChessboardCorners(depth_image, (1,1), corners_refined[center_corner_idx, :, :], checkerboard_found) # 在图像上绘制中心点
        cv2.imshow('Calibration', vis)
        cv2.imwrite('%06d.png' % len(measured_pts), vis)
        cv2.waitKey(1000) # 显示图像1s
        # while cv2.waitKey(1) != ord('c'):
        #     pass
        cv2.destroyAllWindows()
    # if calib_pt_idx == 3:
    #     break

# 机器人回到标定起始点
robot.MoveL(calib_home_ponit, Override=Override)
# 坐标数据转换
observed_pix = np.asarray(observed_pix) # 像素坐标系下的坐标值
observed_pts = np.asarray(observed_pts) # 相机坐标系下的坐标值
measured_pts = np.asarray(measured_pts) # 机器人坐标系下的坐标值
robot2camera = np.eye(4) # 机器人坐标到相机坐标系的变换矩阵

## 2.4 计算实际坐标系到相机坐标系的变换误差
# 利用SVD估计刚体变换（from Nghia Ho）
def get_rigid_transform(A, B):
    """
    计算两组点之间的最优刚体变换（旋转和平移），使得第一组点尽可能地与第二组点对齐，返回旋转矩阵和平移向量
    刚体变换意味着变换后的物体保持形状和大小不变，只发生旋转和平移
    """
    assert len(A) == len(B) # 两组点的数量必须相等
    N = A.shape[0] # N为点的数量，坐标的维度为3维
    centroid_A = np.mean(A, axis=0) # 计算A的质心，形状为(3,)，axis=0表示按列求均值
    centroid_B = np.mean(B, axis=0) # 计算B的质心
    AA = A - np.tile(centroid_A, (N, 1)) # 中心化点集A，tile函数将centroid_A重复N次，变为N行3列
    BB = B - np.tile(centroid_B, (N, 1)) # 中心化点集B
    H = np.dot(np.transpose(AA), BB) # 计算交叉协方差矩阵
    U, S, Vt = np.linalg.svd(H) # 奇异值分解
    R = np.dot(Vt.T, U.T) # 计算旋转矩阵
    if np.linalg.det(R) < 0: # 处理特殊的反射情况
       Vt[2, :] *= -1
       R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T # 计算平移向量
    return R, t # 返回旋转矩阵和平移向量

# 计算刚体变换误差
def get_rigid_transform_error(z_scale):
    global measured_pts, observed_pts, observed_pix, robot2camera
    # 应用Z轴偏移并使用相机内参计算新的观测点
    # observed_z = observed_pts[:, 2:] * z_scale # Z轴缩放
    # observed_x = np.multiply(observed_pix[:, [0], None] - robot.cam_intrinsics[0][2], observed_z / robot.cam_intrinsics[0][0]) # 相机内参矫正，不加None会导致广播过程中维度错误，应该是Numpy版本的问题，当前Numpy=1.23.5，换版本后要注意报错
    # observed_y = np.multiply(observed_pix[:, [1], None] - robot.cam_intrinsics[1][2], observed_z / robot.cam_intrinsics[1][1])
    # new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1) # 新的观测点
    # new_observed_pts = np.squeeze(new_observed_pts) # 去除维度为1的维度，因为上面的广播会增加维度，变为(N, 3)
    new_observed_pts = observed_pts.copy()
    new_observed_pts[:, 2:] *= z_scale # Z轴缩放
    # 在测量点和新观测点之间估计刚体变换（实际坐标到相机坐标）
    R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts)) # PS：这里的R和t是实际坐标到相机坐标的变换
    t.shape = (3, 1)
    robot2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0) # 组合R和t，并在底部添加[0, 0, 0, 1]，变为为4x4的齐次变换矩阵
    # 计算刚体变换的均方误差RMSE
    registered_pts = np.dot(R, np.transpose(measured_pts)) + np.tile(t, (1, measured_pts.shape[0])) # 先对机器人坐标系下的坐标进行旋转变换，然后加上平移向量，得到在相机坐标系下的坐标
    error = np.transpose(registered_pts) - new_observed_pts
    error = np.sum(np.multiply(error, error))
    rmse = np.sqrt(error / measured_pts.shape[0])
    return rmse

## 2.5 最优化Z轴缩放
print('Calibrating...')
z_scale_init = 1 # Z轴缩放初始值
print(f"z_scale_init: {z_scale_init}, get_rigid_transform_error: {get_rigid_transform_error(z_scale_init)}")
optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead') # Nelder-Mead最优化方法，最小化刚体变换误差，获得最优Z轴缩放系数
camera_depth_offset = optim_result.x # 最优Z轴缩放系数
print(f"camera_depth_offset: {camera_depth_offset}, get_rigid_transform_error: {optim_result.fun}") # optim_result.fun为最小化的误差
## 2.6 保存相机优化偏移和相机姿态
print('Saving...')
np.savetxt('camera_depth_scale.txt', camera_depth_offset, delimiter=' ') #（======== 修改 9 ========）
get_rigid_transform_error(camera_depth_offset)
camera_pose = np.linalg.inv(robot2camera) # 求逆矩阵，即相机测量坐标到实际坐标的变换矩阵
np.savetxt('camera_pose.txt', camera_pose, delimiter=' ') #（======== 修改 9 ========）
print('Calibration finished.')