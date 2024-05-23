#!/usr/bin/env python
import cv2
from Gemini_camera import OrbbecGeminiCamera as Camera
import time

# 初始化相机
camera = Camera(device_id='AY3C731008G',
                color_width=1920,
                color_height=1080,
                depth_width=1280,
                depth_height=800
                )
camera.depth_mode(0)
camera.connect()
camera.ldp_switch(False) # 关闭LDP
# cam.laser_switch(True) # 打开激光
camera.soft_filter_switch(True) # 打开软件滤波
save_dir = "../photo"
# 获取图像
while True:
    image = camera.get_image_bundle()
    color_image = image['rgb']
    depth_image = image['aligned_depth'] / 1000.0
    camera.depth = depth_image

    # 图像显示
    cv2.namedWindow('Depth') # 创建图像窗口
    cv2.setMouseCallback('Depth', camera.mouseclick_callback) # 设置鼠标回调函数，Depth为窗口名，mouseclick_callback为回调函数
    pixel_x, pixel_y = 10, 30 # 设置显示坐标的位置
    cv2.putText(depth_image, camera.depth_str, (pixel_x, pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # 0.5为字体大小，(0, 255, 0)为颜色，1为字体粗细
    cv2.imshow('RGB', color_image)
    cv2.imshow('Depth', depth_image)
    key = cv2.waitKey(30)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    rgb_name = save_dir + '/rgb_' + str(time_str) + '.png'
    depth_name = save_dir + '/depth_' + str(time_str) + '.tiff'
    filtered_depth_name = save_dir + '/depth_inpaint_' + str(time_str) + '.tiff'
    points_name = save_dir + '/points_' + str(time_str) + '.txt'
    if key & 0xFF == ord('s'):
        # camera.save_photo(rgb_name=rgb_name, depth_name=depth_name, filtered_depth_name=filtered_depth_name, points_name=points_name)
        camera.save_photo(rgb_name=rgb_name, depth_name=depth_name, filtered_depth_name=filtered_depth_name)
        # camera.save_photo(rgb_name=rgb_name, depth_name=depth_name, points_name=points_name)
    if key & 0xFF == ord('c'):
        cv2.destroyAllWindows()
        break
