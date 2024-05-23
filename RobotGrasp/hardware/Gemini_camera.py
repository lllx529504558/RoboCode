"""
=====================================
机器人视觉点动抓取
Robot：Hans Robot - Elfin
Camera：Gemini2
Version：V1.0
Copyright：XMU-GZW
=====================================
主要代码功能：
1. 相机参数初始化 - 确定相机设备ID、RGB图像尺寸、RGB图像帧率、深度图像尺寸、深度图像帧率，初始化RGB内参、深度图内参、深度缩放系数
2. 相机连接 - 配置相机参数、获取相机数据流，确定对齐模式、深度尺度、相机内参，打印相机数据流信息
3. 相机功能函数 - 深度模式选择，开关激光，开关LDP，开关软件滤波，获取相机内参（输出为特定格式），获取RGB图像和深度图像，保存图像
4. 相机图像显示 - RGB与深度图像叠加显示、RGB与深度图像独立显示
5. 相机工具函数 - OpenCV窗口的回调函数，深度图像修复缺失值，深度图像归一化

需要修改部分：
1. py文件路径

版本文件需求：
1. Python 3.8 (强制要求，因为HansRobotSDK只支持Python3.8)
2. Numpy 1.23.5（广播计算问题，版本不同计算方法不同，换版本可能要Debug）
3. OrbbecSDK cp3.8-win-amd64 & utils.py（官方提供）
=====================================
"""

import sys
sys.path.append('../') # 添加上级目录到系统路径中（======== 修改 1 ========）
import cv2
import numpy as np
from imageio import imsave
from utils.Gemini.utils import frame_to_bgr_image
import utils.Gemini.pyorbbecsdk as obc

class OrbbecGeminiCamera:
    """
    Orbbec Gemini2 Camera class

    :param device_id: str, 相机设备ID, 默认值：'AY3C731008G'
    :param color_width: int, RGB图像宽度, 默认值：640
    :param color_height: int, RGB图像高度, 默认值：480
    :param color_fps: int, RGB图像帧率, 默认值：5
    :param depth_width: int, 深度图像宽度, 默认值：640
    :param depth_height: int, 深度图像高度, 默认值：400
    :param depth_fps: int, 深度图像帧率 默认值：5

    Gemini2相机可获取两个数据流，一个是RGB图像，一个是深度图像，其可获取格式如下：
    Color sensor: 1920*1080, 1280*720, 640*480, 640*360
    Color frame: 30, 15, 10, 5
    Color image format: MJPG, YUYV, RGB
    Depth sensor: 1280*800, 640*400, 320*200
    Depth frame: 30, 15, 10, 5
    Depth image format: Y16, Y14, RLE
    """
    ## 1. 相机参数初始化
    def __init__(self,
                 device_id='AY3C731008G',
                 color_width=640,
                 color_height=480,
                 color_fps=5,
                 depth_width=640,
                 depth_height=400,
                 depth_fps=5):
        self.device_id = device_id
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps

        self.pipeline = None
        self.config = None
        self.rgb_intrinsic = None
        self.rgb_distortion = None
        self.depth_intrinsic = None
        self.depth_distortion = None
        self.transform = None
        self.depth_scale = None
        self.depth = None
        self.depth_str = None # 初始化用于在图像上显示的机器人坐标系下的坐标字符串
        self.context = obc.Context() # 创建上下文
        device_list = self.context.query_devices() # 查询设备
        curr_device_cnt = device_list.get_count() # 获取设备数量
        print(f"Total find {curr_device_cnt} device connected") # 打印设备数量
        self.device = self.context.query_devices().get_device_by_index(0) # 获取设备
    
    ## 2. 相机连接
    def connect(self, window_size=None):
        # 配置相机参数
        self.pipeline = obc.Pipeline(self.device)
        self.config = obc.Config()
        # RGB与深度图合并模式打开，选择SW（硬件对齐模式）
        self.config.set_align_mode(obc.OBAlignMode.SW_MODE) # 设置对齐模式
        self.pipeline.enable_frame_sync() # 启用帧同步，好像没什么用，只要设置了模型，不开启帧同步也会对齐
        # 获取RGB图像
        profile_color = self.pipeline.get_stream_profile_list(obc.OBSensorType.COLOR_SENSOR)
        color_profile = profile_color.get_video_stream_profile(self.color_width, self.color_height, obc.OBFormat.RGB, self.color_fps)
        self.config.enable_stream(color_profile)
        # 获取深度图像
        profile_depth = self.pipeline.get_stream_profile_list(obc.OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_depth.get_video_stream_profile(self.depth_width, self.depth_height, obc.OBFormat.Y16, self.depth_fps)
        self.config.enable_stream(depth_profile)
        # 确定深度尺度
        self.pipeline.start(self.config) # 启动相机
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue
            else:
                self.depth_scale = depth_frame.get_depth_scale()
                break
        # 确定相机内参
        self.camera_param = self.pipeline.get_camera_param()
        self.rgb_intrinsic = self.camera_param.rgb_intrinsic
        self.depth_intrinsic = self.camera_param.depth_intrinsic
        self.rgb_distortion = self.camera_param.rgb_distortion
        self.depth_distortion = self.camera_param.depth_distortion
        self.transform = self.camera_param.transform
        # 打印相机数据流信息
        print(f"color profile : WxH={color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps_{color_profile.get_format()}")                                
        print(f"depth profile : WxH={depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps_{depth_profile.get_format()}")
    
    ## 3. 相机功能函数
    # 相机深度模式选择
    def depth_mode(self, depth_work_mode=None):
        """
        相机深度模式选择
        :param depth_work_mode: int, 深度模式选择，0-3
            PS: 如果不指定深度模式，将打印当前深度模式

        深度模式介绍：
        0：Unbinned Dense Default 精度和质量优先
        1：Binned Sparse Default 低功耗，小盲区，高帧率
        2：Unbinned Sparse Default 平衡质量与功耗，提升低反和半室外效果
        3：Obstacle Avoidance 防撞模式（不理解，不建议使用）
        """
        if depth_work_mode is None:
            current_depth_work_mode = self.device.get_depth_work_mode()
            print("Current depth work mode: ", current_depth_work_mode)
            return
        depth_work_mode_list = self.device.get_depth_work_mode_list()
        depth_work_mode = depth_work_mode_list.get_depth_work_mode_by_index(depth_work_mode)
        self.device.set_depth_work_mode(depth_work_mode)
        print("Change depth mode to:", depth_work_mode)

    # 开关激光
    def laser_switch(self, laser_switch):
        """
        开关激光
        :param laser_switch: bool, 激光开关，True-打开，False-关闭
        """
        self.device.set_bool_property(obc.OBPropertyID.OB_PROP_LASER_BOOL, laser_switch)
        print("Change laser state to:", "Open" if laser_switch else "Close")
    
    # 开关LDP
    def ldp_switch(self, ldp_switch):
        """
        开关LDP
        :param ldp_switch: bool, LDP开关，True-打开，False-关闭
        """
        self.device.set_bool_property(obc.OBPropertyID.OB_PROP_LDP_BOOL, ldp_switch)
        print("Change LDP state to:", "Open" if ldp_switch else "Close")
    
    # 开关软件滤波
    def soft_filter_switch(self, software_filter_switch):
        """
        开关软件滤波
        :param software_filter_switch: bool, 软件滤波开关，True-打开，False-关闭
        """
        self.device.set_bool_property(obc.OBPropertyID.OB_PROP_DEPTH_SOFT_FILTER_BOOL, software_filter_switch)
        print("Change software filter state to:", "Open" if software_filter_switch else "Close")

    # 获取相机内参
    def get_intrinsics(self):
        raw_intrinsics = self.rgb_intrinsic
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([512.413, 0, 322.734, 0, 512.491, 234.74, 0, 0, 1]).reshape(3,3) #640 480
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.cx, 0, raw_intrinsics.fy, raw_intrinsics.cy, 0, 0, 1]).reshape(3, 3)
        print("intrinsics:\n", intrinsics)
        return intrinsics

    # 获取RGB图像和深度图像
    def get_image_bundle(self):
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if color_frame is None or depth_frame is None:
                continue
            else:
                break
        color_image = frame_to_bgr_image(color_frame) # RGB --> BGR
        depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16) # Depth --> uint16
        depth_image = depth_image.reshape((self.color_height, self.color_width)) # 重置深度图像尺寸
        depth_image = depth_image.astype(np.float32) * self.depth_scale # 调整深度图偏移量，与RGB图像对齐
        depth_image = self.inpaint(depth_image) # 补全缺失值
        depth_image = np.expand_dims(depth_image, axis=2) # 增加深度图像通道，方便后续与RGB图合并
        return {
                'rgb': color_image,
                'aligned_depth': depth_image,
                }
    
    # 保存图像（RGB、Depth、Points）
    def save_photo(self, rgb_name=None, depth_name=None, filtered_depth_name=None, points_name=None):
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if color_frame is None or depth_frame is None:
                continue
            else:
                break
        # 保存点云数据txt
        if points_name is not None:
            points = frames.convert_to_color_points(self.camera_param)
            with open(points_name, "w") as f:
                f.write("PCD\n")
                f.write("FIELDS x y z r g b index\n")
                f.write("POINTS {}\n".format(len(points)))
                f.write("DATA ascii\n")
                for point in points:
                    f.write(
                        "{} {} {} {} {} {}\n".format(point.x, point.y, point.z, point.r, point.g, point.b))
        # 获取RGB和深度图像
        color_image = frame_to_bgr_image(color_frame) # RGB --> BGR
        depth_image = np.frombuffer(depth_frame.get_data(), dtype=np.uint16) # Depth --> uint16
        print("depth_image:", depth_image)
        depth_image = depth_image / 1000 # 深度图像单位转换为m，否则，保存后为全白图像；当然，不转换也可以，不影响深度值保存
        print("depth_image_m:", depth_image)
        depth_image = depth_image.reshape((self.color_height, self.color_width)) # 重置深度图像尺寸
        depth_image = depth_image.astype(np.float32) * self.depth_scale # 调整深度图偏移量，与RGB图像对齐
        # 保存png和tiff图像
        if rgb_name is not None:
            cv2.imwrite(rgb_name, color_image)
        if depth_name is not None:
            imsave(depth_name, depth_image)
        if filtered_depth_name is not None:
            filtered_depth_image = self.inpaint(depth_image) # 深度图像修复缺失值
            imsave(filtered_depth_name, filtered_depth_image)
        print("Save photo successfully!")
        
    ## 4. 相机图像显示
    def plot_image_bundle(self, mode='overlay'):
        if mode == 'overlay':
            # RGB和深度图像叠加显示
            image = self.get_image_bundle()
            rgb = image['rgb']
            depth = image['aligned_depth']
            self.depth = depth
            # 由于深度图的值可能非常大，直接显示可能会看不到任何细节，所以需要将深度图的值归一化到0-255的范围内，并转换为8位无符号整型
            depth_image = self.normalize_depth(depth) # 深度图像归一化
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET) # 将归一化后的深度图转换为伪彩色图像，以更好地观察深度变化
            # 重合彩色图像与深度图像
            depth_image = cv2.addWeighted(rgb, 0.5, depth_image, 0.5, 0)
            # 图像显示
            cv2.namedWindow('RGB & Depth') # 创建图像窗口
            cv2.setMouseCallback('RGB & Depth', self.mouseclick_callback) # 设置鼠标回调函数，RGB & Depth为窗口名，mouseclick_callback为回调函数
            pixel_x, pixel_y = 10, 30 # 设置显示坐标的位置
            cv2.putText(depth_image, self.depth_str, (pixel_x, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # 0.5为字体大小，(0, 255, 0)为颜色，1为字体粗细
            cv2.imshow('RGB & Depth', depth_image) # 显示图像
        elif mode == 'split':
            # 使用OPENCV，RGB和深度图像分开显示
            image = self.get_image_bundle()
            rgb = image['rgb']
            depth = image['aligned_depth']
            self.depth = depth
            depth_image = self.normalize_depth(depth) # 深度图像归一化
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET) # 将归一化后的深度图转换为伪彩色图像，以更好地观察深度变化
            # 图像显示
            cv2.namedWindow('Depth') # 创建图像窗口
            cv2.setMouseCallback('Depth', self.mouseclick_callback) # 设置鼠标回调函数，RGB & Depth为窗口名，mouseclick_callback为回调函数
            pixel_x, pixel_y = 10, 30 # 设置显示坐标的位置
            cv2.putText(depth_image, self.depth_str, (pixel_x, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # 0.5为字体大小，(0, 255, 0)为颜色，1为字体粗细
            cv2.imshow('RGB', rgb)
            cv2.imshow('Depth', depth_image)
    
    ## 5. 相机工具函数
    # OpenCV窗口的回调函数
    def mouseclick_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: # 判断鼠标左键是否按下
            click_point_pix = (x, y)
            # 获取手动选择的像素点在相机坐标系下的坐标
            click_z = self.depth[click_point_pix[1]][click_point_pix[0]] # 获取深度图像中的深度值
            self.depth_str = f"Select Pixel Depth is {click_z[0]} mm" # 用于在图像上显示选择点深度值的字符串
    
    # 深度图像修复函数
    def inpaint(self, img, missing_value=0):
        """
        修复深度图像中的缺失值
        :param img: 输入的深度图像
        :param missing_value: 缺失位置的值，默认为0，即对应深度值为0的位置为缺失值
        """
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == missing_value).astype(np.uint8)
        # OpenCV要求将像素值缩放在 -1~1 范围内，且为浮点数
        scale = np.abs(img).max()
        img = img.astype(np.float32) / scale  # 必须是32位浮点数，64位浮点数会报错
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
        # 回到原始大小和值范围
        img = img[1:-1, 1:-1]
        img = img * scale
        return img
    
    # 深度图像归一化
    def normalize_depth(self, frame): # 归一化深度图像
        # 使用全局最大值进行归一化
        global_max = 1500
        # 归一化深度图像
        normalized_frame = np.clip(frame / global_max * 255, 0, 255)
        return normalized_frame.astype(np.uint8)

# 主函数
if __name__ == '__main__':
    cam = OrbbecGeminiCamera(device_id='AY3C731008G',
                             color_width=1920,
                             color_height=1080,
                             depth_width=1280,
                             depth_height=800)
    cam.depth_mode(0)
    cam.connect()
    cam.ldp_switch(False) # 关闭LDP
    # cam.laser_switch(True) # 打开激光
    cam.soft_filter_switch(True) # 打开软件滤波
    while True:
        cam.plot_image_bundle(mode='overlay')
        # cam.plot_image_bundle(mode='split')
        # 按下'c'键关闭窗口
        if cv2.waitKey(1) == ord('c'):
            cv2.destroyAllWindows()
            break


