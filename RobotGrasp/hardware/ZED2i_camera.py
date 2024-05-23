import pyzed.sl as sl
import cv2
import numpy as np
from skimage.io import imsave
import struct

class ZED2iCamera:
    """
    ZED2i相机类 用于获取ZED2i相机的参数和图像

    :param device_id: 相机设备ID
    :param image_resolution: 图像分辨率
    :param image_fps: 图像帧率

    ZED2i相机参数可获取2个数据流，分别为左右相机图像
    VIDEO MODE: HD1080, HD720, HD2K, VGA
    FRAME RATE (FPS): HD2K(4416x1242) - 15
                      HD1080(3840x1080) - 15, 30
                      HD720(2560x720) - 15, 30, 60
                      VGA(1344x376) - 15, 30, 60, 100
    DEPTH MODE: ULTRA（为基于计算机视觉的技术提供最高的深度范围和沿感应范围更好地保持 Z 轴精度）
                PERFORMANCE（设计流畅，可能会遗漏一些细节）
                NEURAL、NEURAL+（使用 AI 技术将深度传感提升到一个新的精度水平。即使在最具挑战性的情况下，也能准确而流畅）
                QUALITY（具有强大的过滤阶段，可提供光滑的表面）

    """
    ## 1. 相机参数初始化
    def __init__(self,
                 device_id='34680631',
                 image_resolution='HD1080',
                 image_fps=15,
                 ):
        # 创建相机对象
        self.zed = sl.Camera()
        # 获取设备列表
        self.device_list = self.zed.get_device_list()
        self.camera_id = device_id
        self.depth_str = None # 初始化用于在图像上显示的机器人坐标系下的坐标字符串
        # 设置初始化参数
        self.init_params = sl.InitParameters()
        self.init_params.sdk_verbose = 0
        self.init_params.camera_fps = image_fps
        # self.init_params.camera_disable_self_calib = True
        self.resolution_mode(image_resolution)
        
       
        # 输出相机参数
        print(f"Total find {len(self.device_list)} device connected") # 打印设备数量
        
    
    ## 2. 相机连接
    def connect(self, enable_fill_mode=False,
                depth_mode='ULTRA',
                depth_unit='mm',
                depth_min=300,
                depth_max=2000):
        # 设置深度值单位和范围
        self.unit = depth_unit
        self.depth_min = depth_min
        self.depth_max = depth_max
        # 设置相机参数
        self.runtime_parameters = sl.RuntimeParameters(enable_fill_mode=enable_fill_mode) # 启用填充模式
        self.depth_range(unit=self.unit, min=self.depth_min, max=self.depth_max)
        self.depth_mode(depth_mode)
        # 打开相机
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Camera Open : "+repr(err)+". Exit program.")
            exit(-1)
        print("Camera connected")
        # 获取相机参数
        zed_info = self.zed.get_camera_information()
        self.camera_info = zed_info.camera_model
        self.image_width = zed_info.camera_configuration.resolution.width
        self.image_height = zed_info.camera_configuration.resolution.height
        self.intrinsic_fx = zed_info.camera_configuration.calibration_parameters.left_cam.fx # 左右相机内参相同
        self.intrinsic_fy = zed_info.camera_configuration.calibration_parameters.left_cam.fy
        self.intrinsic_cx = zed_info.camera_configuration.calibration_parameters.left_cam.cx
        self.intrinsic_cy = zed_info.camera_configuration.calibration_parameters.left_cam.cy
        self.baseline = zed_info.camera_configuration.calibration_parameters.get_camera_baseline()
        self.image_distortion = zed_info.camera_configuration.calibration_parameters.left_cam.disto
        self.transform = zed_info.camera_configuration.calibration_parameters.stereo_transform.m
        # 打印相机数据流信息
        print(f"Camera model: {self.camera_info}") # 打印相机型号
        print(f"Image profile: WxH={self.image_width}x{self.image_height}@{self.init_params.camera_fps}fps") # 打印图像分辨率和帧率
    
    # 关闭相机 
    def disconnect(self):
        self.zed.close()
        print("Camera disconnected")
    
    ## 3. 相机功能函数
    # 相机分辨率选择
    def resolution_mode(self, image_resolution=None):
        """
        :param image_resolution: 图像分辨率
        """
        if image_resolution is None:
            print("Current resolution mode: HD1080")
            return
        elif image_resolution == 'HD1080':
            self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif image_resolution == 'HD720':
            self.init_params.camera_resolution = sl.RESOLUTION.HD720
        elif image_resolution == 'HD2K':
            self.init_params.camera_resolution = sl.RESOLUTION.HD2K
        elif image_resolution == 'VGA':
            self.init_params.camera_resolution = sl.RESOLUTION.VGA

    # 相机深度模式选择
    def depth_mode(self, depth_work_mode=None):
        """
        :param depth_work_mode: 深度模式
            'ULTRA'（为基于计算机视觉的技术提供最高的深度范围和沿感应范围更好地保持 Z 轴精度）
            'PERFORMANCE'（设计流畅，可能会遗漏一些细节）
            'NEURAL'、'NEURAL+'（使用 AI 技术将深度传感提升到一个新的精度水平）
            'QUALITY'（具有强大的过滤阶段，可提供光滑的表面）
        """
        if depth_work_mode is None:
            print("Current depth work mode: ULTRA")
            return
        elif depth_work_mode == 'PERFORMANCE':
            self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        elif depth_work_mode == 'NEURAL':
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        elif depth_work_mode == 'NEURAL+':
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
        elif depth_work_mode == 'ULTRA':
            self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        elif depth_work_mode == 'QUALITY':
            self.init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        print("Change depth mode to:", depth_work_mode)
    
    # 深度识别范围
    def depth_range(self, unit='mm', min=None, max=None):
        """
        :param unit: 深度值单位，可选'mm'、'cm'、'm'，默认为'mm'
        """
        if unit == 'mm':
            self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        elif unit == 'cm':
            self.init_params.coordinate_units = sl.UNIT.CENTIMETER
        elif unit == 'm':
            self.init_params.coordinate_units = sl.UNIT.METER
        if min is not None:
            self.init_params.depth_minimum_distance = min
        if max is not None:
            self.init_params.depth_maximum_distance = max
        print(f"Change depth range to: min_{min}{unit}, max_{max}{unit}")
    
    # 深度图像滤波
    def depth_filter(self, confidence_threshold=None, texture_confidence_threshold=None):
        confidence_map = sl.Mat()
        if confidence_threshold is not None:
            self.runtime_parameters.confidence_threshold = confidence_threshold # 删除边缘上的点，以避免出现“链接”对象
        elif texture_confidence_threshold is not None:
            self.runtime_parameters.texture_confidence_threshold = texture_confidence_threshold # 删除低纹理区域的点，即图像均匀区域的点
        elif confidence_threshold is not None and texture_confidence_threshold is not None:
            print("Please input confidence_threshold OR texture_confidence_threshold")
            exit(-1)
        self.zed.retrieve_measure(confidence_map, sl.MEASURE.CONFIDENCE)
        return confidence_map.get_data() # 返回深度图像滤波后的数据，格式为numpy数组
    
    # 获取相机内参
    def get_intrinsics(self):
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        intrinsics = np.array([self.intrinsic_fx, 0, self.intrinsic_cx, 0, self.intrinsic_fy, self.intrinsic_cy, 0, 0, 1]).reshape(3, 3)
        print("intrinsics:\n", intrinsics)
        return intrinsics
    
    # 获取相机传感器温度
    def get_temperature(self):
        sensors_data = sl.SensorsData()
        if self.zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
            # 这里分别拿到左、右相机，imu，气压计四者的温度
            temperature_left = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.ONBOARD_LEFT)
            temperature_right = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.ONBOARD_RIGHT)
            temperature_imu = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.IMU)
            temperature_barometer = sensors_data.get_temperature_data().get(sl.SENSOR_LOCATION.BAROMETER)
            print("Left: {:.2f}, Right: {:.2f}, IMU: {:.2f}, Barometer: {:.2f}\r\n".format(temperature_left,
                                                                                           temperature_right,
                                                                                           temperature_imu,
                                                                                           temperature_barometer))
    
    # 获取RGB图像和深度图像
    def get_image_bundle(self, inpaint=False):
        image = sl.Mat()
        depth = sl.Mat()
        grab_index = 0
        while True:
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT) # 获取左相机图像
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # 获取深度值
                break
            elif grab_index == 5:
                print("Grab image time out!")
                break
            else:
                print("Grab image failed!")
                grab_index += 1
                continue
        color_image = image.get_data().copy()[:, :, :3] # BGRA --> BGR，ZED默认获取BGR图像
        depth_image = depth.get_data().copy()
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        if inpaint:
            depth_image = self.inpaint(depth_image) # 补全缺失值
        depth_image = np.expand_dims(depth_image, axis=2) # 增加深度图像通道，方便后续与RGB图合并
        return {
                'rgb': color_image,
                'aligned_depth': depth_image,
                }
    
    # 保存图像（RGB、Depth、Points）
    def save_photo(self, rgb_name=None, depth_name=None, points_name=None, inpaint=False):
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()
        grab_index = 0
        while True:
            if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(image, sl.VIEW.LEFT) # 获取左相机图像
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH) # 获取深度值
                break
            elif grab_index == 5:
                print("Grab image time out!")
                break
            else:
                print("Grab image failed!")
                grab_index += 1
                continue
        # 保存点云数据txt
        if points_name is not None:
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            points = point_cloud.get_data()
            with open(points_name, "w") as f:
                index = 0
                f.write("PCD\n")
                f.write("FIELDS x y z r g b index\n")
                f.write("POINTS {}\n".format(points.shape[0]*points.shape[1]))
                f.write("DATA ascii\n")
                for i in range(points.shape[0]):
                    for j in range(points.shape[1]):
                        x, y, z, rgba = points[i, j]
                        rgba = np.nan_to_num(rgba, nan=0)
                        r, g, b, a = self.unpack_color(rgba)
                        # 写入文件，格式为 X, Y, Z, R, G, B, index
                        f.write(f"{x} {y} {z} {r} {g} {b} {index}\n")
                        index += 1
            print("Save point cloud successfully!")

        # 获取RGB和深度图像
        color_image = image.get_data().copy()[:, :, :3] # BGRA --> BGR，ZED默认获取BGR图像
        depth_image = depth.get_data().copy()
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        if inpaint:
            depth_image = self.inpaint(depth_image) # 补全缺失值
        # 保存png和tiff图像
        if rgb_name is not None:
            cv2.imwrite(rgb_name, color_image)
        if depth_name is not None:
            imsave(depth_name, depth_image)
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
            depth_image = cv2.addWeighted(rgb, 0.4, depth_image, 0.6, 0)
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
            self.depth_str = f"Select Pixel {click_point_pix}, Depth is {click_z[0]} {self.unit}" # 用于在图像上显示选择点深度值的字符串
    
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
        # scale = np.abs(img).max()
        scale = 2000
        img = img.astype(np.float32) / scale  # 必须是32位浮点数，64位浮点数会报错
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
        # 回到原始大小和值范围
        img = img[1:-1, 1:-1]
        img = img * scale
        return img
    
    # 深度图像归一化
    def normalize_depth(self, frame): # 使用全局最大值进行归一化
        # 归一化深度图像
        normalized_frame = np.clip(frame / self.depth_max * 255, 0, 255)
        return normalized_frame.astype(np.uint8)
    
    # 将32位浮点颜色值转换为R, G, B, A整数值
    def unpack_color(self, float_color):
        packed_color = struct.pack('f', float_color)
        int_color = struct.unpack('I', packed_color)[0]
        r = (int_color >> 0) & 0xFF
        g = (int_color >> 8) & 0xFF
        b = (int_color >> 16) & 0xFF
        a = (int_color >> 24) & 0xFF
        return r, g, b, a
    

# 主函数
if __name__ == '__main__':
    cam = ZED2iCamera(device_id='34680631',
                      image_resolution='HD1080',
                      image_fps=5)
    cam.connect(enable_fill_mode=True,
                # depth_mode='ULTRA',
                # depth_mode='QUALITY',
                # depth_mode='NEURAL',
                depth_mode='NEURAL+',
                depth_unit='mm',
                depth_min=300,
                depth_max=1500)
    cam.get_intrinsics()
    while True:
        # cam.plot_image_bundle(mode='overlay')
        cam.plot_image_bundle(mode='split')
        # 按下'c'键关闭窗口
        if cv2.waitKey(1) == ord('c'):
            cv2.destroyAllWindows()
            break