"""
=====================================
Hans Robot 封装函数库
Robot：Hans Robot - Elfin
Camera：Gemini2
Version：V1.0
Copyright：XMU-GZW
=====================================
主要代码功能：
1. 机器人参数初始化：
    1.1 连接参数：IP和端口、电箱ID、机器人ID、末端执行器ID、是否使用末端执行器、是否使用相机
    1.2 工作参数：工作空间限制、关节限制、最大速度、最大加速度、TCP坐标、机器人速度比例
    1.3 运动参数：位置容差阈值、回零位置、放置位置、抓取位姿
    1.4 相机参数：相机内参、相机深度尺度
2. 机器人基础函数 - 连接、使能、下电、回零、停止、暂停、继续、复位
3. 机器人工具函数 - 列表字符串转浮点数、弧度转角度、角度转弧度
4. 机器人功能函数 - 启动机器人、重启机器人、获取相机数据、末端执行器检测、末端执行器打开、末端执行器闭合、机器人运动速度设置、机器人运动状态检测、机器人当前状态检测、机器人当前位置获取、机器人移动精度检测、判断机器人是否在工作空间内
5. 机器人运动函数 - 关节点动调节、直线点动调节、路点点动调节、关节运动、直线轨迹运动、圆弧轨迹运动、机器人高精度移动函数
6. 机器人抓取函数 - 机器人平面抓取物体、机器人抓取放置物体、机器人推动物体

版本文件需求：
1. Python 3.8 (强制要求，因为HansRobotSDK只支持Python3.8)
2. Numpy 1.23.5（广播计算问题，版本不同计算方法不同，换版本可能要Debug）
3. OrbbecSDK cp3.8-win-amd64 & utils.py（官方提供）
4. Gemini_camera.py（Hans_Robot.py中已集成，XMU-GZW）
5. HansRobotSDK V1.0.0.3（官方提供）
=====================================
"""

#!/usr/bin/env python
from utils.HansRobot.CPS import CPSClient
# from hardware.Gemini_camera import OrbbecGeminiCamera as Camera
from hardware.ZED2i_camera import ZED2iCamera as Camera
import time
import copy
import numpy as np

class HansRobot():
    ## 1. 机器人参数初始化
    def __init__(self, tcp_host_ip='192.168.4.40', tcp_port=10003,
                 boxID=0, rbtID=0, endID=0, Override=0.2, TcpName="TCP",
                 place_position = [550, -400, 20, 180, 0, -160],
                 workspace_limits=None, joint_limits=None,
                 is_use_effector=True, is_use_camera=False):
        """
        初始化机器人参数
        :param tcp_host_ip: 机器人IP地址
        :param tcp_port: 机器人端口
        :param boxID: 电箱ID
        :param rbtID: 机器人ID
        :param endID: 末端执行器ID
        :param Override: 机器人速度比例
        :param place_position: 机器人放置位置 [Type: list]->[X, Y, Z]
        :param workspace_limits: 机器人工作空间限制 [Type: list]->[[X_max, Y_max, Z_max], [X_min, Y_min, Z_min], [X, Y, Z, RX, RY, RZ]]
        :param joint_limits: 机器人关节限制 [Type: list]->[[J1_max, J2_max, J3_max, J4_max, J5_max, J6_max], [J1_min, J2_min, J3_min, J4_min, J5_min, J6_min]]
        :param is_use_effector: 是否使用末端执行器
        :param is_use_camera: 是否使用相机
        """
        # 初始化机器人连接参数
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.is_use_effector = is_use_effector
        self.is_use_camera = is_use_camera
        # 初始化机器人工作参数
        self.cps = CPSClient()
        self.boxID = boxID
        self.rbtID = rbtID
        self.endID = endID
        self.Override = Override
        self.TcpName = TcpName
        self.joint_vel = 180 # 关节速度最大值，超过不运行
        self.joint_acc = 360 # 关节加速度最大值，超过不运行
        self.linear_vel = 3000 # 直线速度实际最大值，超过也可运行，但速度不增加
        self.linear_acc = 2500 # 直线加速度实际最大值，超过也可运行，但速度不增加
        self.cps.HRIF_SetTCPByName(self.boxID, self.rbtID, self.TcpName)
        # 初始化机器人工作区间
        if workspace_limits is None:
            workspace_min_limits = [-600, -400, 100] # X, Y, Z
            workspace_max_limits = [600, 200, 500]
            workspace_ucs_limits = [0, 0, 0, 0, 0, 0]
            self.workspace_limits = [workspace_max_limits, workspace_min_limits, workspace_ucs_limits]
        else:
            self.workspace_limits = workspace_limits
        if joint_limits is None:
            joint_min_limits = [-360, -135, -153, -360, -180, -360] # J1, J2, J3, J4, J5, J6
            joint_max_limits = [360, 135, 153, 360, 180, 360]
            self.joint_limits = [joint_max_limits, joint_min_limits]
        else:
            self.joint_limits = joint_limits
        self.cps.HRIF_SetMaxPcsRange(self.boxID, self.rbtID, self.workspace_limits[0], self.workspace_limits[1], self.workspace_limits[2])
        self.cps.HRIF_SetMaxAcsRange(self.boxID, self.rbtID, self.joint_limits[0], self.joint_limits[1])

        # 位置容差阈值
        self.pos_tolerance = [0.003, 0.003, 0.003, 0.01, 0.01, 0.01]
        # 机器人回零位置（关节位置）
        self.home_joint_config = [0, 0, 90, 0, 90, -20]
        # 机器人指定放置位置（笛卡尔位置）
        self.place_position = place_position
        # 机器人末端执行器抓取位姿
        self.target_rpy = [-180, 0, -160]
        
        # Gemini2 Camera configuration
        if(self.is_use_camera):
            # Fetch RGB-D data from Gemini2 camera
            self.camera = Camera()
            # self.camera.connect() # Gemini2相机连接
            self.camera.connect(enable_fill_mode=True, depth_mode='NEURAL+', depth_unit='mm', depth_min=300, depth_max=1500) # ZED2i相机连接
            self.cam_intrinsics = self.camera.get_intrinsics() # 获取相机内参，封装为特定格式
        # 加载相机内参和深度尺度（利用calibrate.py进行手眼标定后获取）
        self.cam_pose = np.loadtxt('\RoboGrasp\Calibrate\camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('\RoboGrasp\Calibrate\camera_depth_scale.txt', delimiter=' ')

    ## 2. 机器人基础函数
    # 连接机器人
    def Connect(self):
        name = []
        self.cps.HRIF_Connect(self.boxID, self.tcp_host_ip, self.tcp_port)
        self.cps.HRIF_ReadRobotModel(self.boxID, self.rbtID, name)
        print(f"Connect to robot successfully! Robot load is {name[0]}.")

    # 断开机器人连接
    def Disconnect(self):
        self.cps.HRIF_DisConnect(self.boxID)
        print("Disconnect from robot successfully!")

    # 机器人使能
    def Enable(self):
        self.cps.HRIF_GrpEnable(self.boxID, self.rbtID)
        print("Enable robot successfully!")

    # 机器人去使能
    def Disable(self):
        # 机器人去使能，PS：机器人运行过程中不能去使能
        self.cps.HRIF_GrpDisable(self.boxID, self.rbtID)
        print("Disable robot successfully!")

    # 机器人上电
    def PowerOn(self):
        self.cps.HRIF_Electrify(self.boxID)
        print("Power on robot successfully!")
    
    # 机器人下电
    def PowerOff(self):
        self.cps.HRIF_Blackout(self.boxID)
        print("Power off robot successfully!")

    # 机器人回零
    def GoHome(self, Override=None):
        if Override is None:
            Override = self.Override
        self.MoveJ(self.home_joint_config, Override=Override)
        print("Home robot successfully!")
    
    # 机器人停止
    def Stop(self):
        self.cps.HRIF_GrpStop(self.boxID, self.rbtID)
        print("Stop robot successfully!")
    
    # 机器人暂停
    def Interrupt(self):
        self.cps.HRIF_GrpInterrupt(self.boxID, self.rbtID)
        print("Interrupt robot successfully!")
    
    # 机器人继续
    def Continue(self):
        self.cps.HRIF_GrpContinue(self.boxID, self.rbtID)
        print("Continue robot successfully!")
    
    # 机器人复位
    def Reset(self):
        self.cps.HRIF_GrpReset(self.boxID, self.rbtID)
        print("Reset robot successfully!")
    
    ## 3. 机器人工具函数
    # 列表字符串转浮点数
    def list_str2float(self, list_str):
        return [float(i) for i in list_str]

    # 弧度转角度
    def rad2deg(self, rad):
        return rad * 180 / np.pi
    
    # 角度转弧度
    def deg2rad(self, deg):
        return deg * np.pi / 180
    
    ## 4. 机器人功能函数
    # 启动机器人
    def Start(self):
        self.Connect()
        if not self.RobotState()["Enable"]:
            self.Enable()
        self.GoHome()
        if self.is_use_effector:
            self.OpenEndEffector()
        print("Start robot successfully!")

    # 重启机器人
    def Restart(self):
        self.Reset()
        if not self.RobotState()["Enable"]:
            self.Enable()
        self.GoHome()
        if self.is_use_effector:
            self.OpenEndEffector()
        print("Restart robot successfully!")
    
    # 获取相机数据 
    def get_camera_data(self):
        images = self.camera.get_image_bundle()
        return images['rgb'], images['aligned_depth']

    # 末端执行器检测
    def IsEndEffectorClose(self):
        # 末端执行器是否关闭，1/True：关闭；0/Fasle：打开
        result = []
        self.cps.HRIF_ReadEndDO(self.boxID, self.rbtID, self.endID, result)
        if result[0] == '0':
            return False
        else:
            return True
    
    # 末端执行器打开
    def OpenEndEffector(self):
        # 打开夹爪
        if self.IsEndEffectorClose():
            self.cps.HRIF_SetEndDO(self.boxID, self.rbtID, self.endID, 0)
    
    # 末端执行器闭合
    def CloseEndEffector(self):
        # 闭合夹爪
        if not self.IsEndEffectorClose():
            self.cps.HRIF_SetEndDO(self.boxID, self.rbtID, self.endID, 1)
    
    # 机器人移动速度设置
    def SetOverride(self, Override=None):
        # 设置机器人速度，范围：0~1
        # 仅对点动调节有效，对程序调节无效
        if Override is None:
            Override = self.Override
        self.cps.HRIF_SetOverride(self.boxID, self.rbtID, Override)
        print(f"Set robot speed to {Override*100}% Max Speed.")
    
    # 机器人运动状态检测
    def IsMotionDone(self):
        # 机器人运动状态检测
        time.sleep(0.1)
        result = []
        self.cps.HRIF_IsMotionDone(self.boxID, self.rbtID, result)
        if result[0]:
            return True
        else:
            return False
    
    # 机器人当前状态检测
    def RobotState(self):
        # 机器人当前状态检测
        result = []
        self.cps.HRIF_ReadRobotState(self.boxID, self.rbtID, result)
        State = {"Motion": result[0], "Enable": result[1], "Power": result[9]}
        return State
    
    # 机器人当前位置获取
    def GetCurrentPosition(self):
        """
        获取当前位置：
        : return Pos: Type: dict
                      Key: "JointPos", "RealPos", "TcpPos", "UcsPos" 
                      Value: Type: list
                      Valus: 关节坐标：[J1, J2, J3, J4, J5, J6]
                             笛卡尔坐标：[X, Y, Z, RX, RY, RZ]
                             TCP坐标：[X, Y, Z, RX, RY, RZ]
                             用户坐标：[X, Y, Z, RX, RY, RZ]
        """
        result = []
        self.cps.HRIF_ReadActPos(self.boxID, self.rbtID, result)
        JointPos = self.list_str2float([result[0], result[1], result[2], result[3], result[4], result[5]])
        RealPos = self.list_str2float([result[6], result[7], result[8], result[9], result[10], result[11]])
        TcpPos = self.list_str2float([result[12], result[13], result[14], result[15], result[16], result[17]])
        UcsPos = self.list_str2float([result[18], result[19], result[20], result[21], result[22], result[23]])
        Pos = {"JointPos": JointPos, "RealPos": RealPos, "TcpPos": TcpPos, "UcsPos": UcsPos}
        return Pos

    # 机器人移动精度检测
    def IsPoseClose(self, target_pose, PosType="RealPos"):
        """
        机器人移动精度检测：
        :param target_pose: 目标位置
        :param PosType: 位置类型，"JointPos"：关节坐标，"RealPos"：笛卡尔坐标，"TcpPos"：TCP坐标，"UcsPos"：用户坐标
        :return: 是否到达目标位置
        """
        # 获取当前位置
        current_pose = self.GetCurrentPosition()[PosType]
        if current_pose[3] == 180 and target_pose[3] == -180:
            current_pose[3] = -180
        elif current_pose[3] == -180 and target_pose[3] == 180:
            current_pose[3] = 180
        current_pose = np.array(current_pose)
        target_pose = np.array(target_pose)
        # 检测机器人移动精度
        if np.abs((target_pose - current_pose) < np.array(self.pos_tolerance)).all():
            print("The robot has reached the target position.")
            return True
        else:
            return False

    # 判断机器人是否在工作空间内
    def IsInWorkspace(self, target_pose, target_rpy=None): ## 坐标系有问题，记得修改，确定TCP和UCS(Base)有什么不同
        """
        判断机器人是否在工作空间内：
        :param target_pose: 目标位置，[X, Y, Z]
        :param target_rpy: 机器人抓取角度，[RX, RY, RZ]
        :return: 是否在工作空间内
        """
        # 判定抓取的位置是否处于工作空间
        for i in range(3):
            min_val = self.workspace_limits[1][i]  # 计算最小值
            max_val = self.workspace_limits[0][i]  # 计算最大值
            # 检查当前元素是否在范围内
            if not (min_val <= target_pose[i] <= max_val):
                print("The robot is out of workspace!")
                return False
        # 获取抓取位姿
        if target_rpy == None:
            target_rpy = self.target_rpy
        else:
            # 判定抓取的角度RPY是否在规定范围内 [-pi,pi]
            for i in range(3):
                target_rpy[i] = self.deg2rad(target_rpy[i])
                if target_rpy[i] > np.pi:
                    target_rpy[i] -= 2 * np.pi
                elif target_rpy[i] < - np.pi:
                    target_rpy[i] += 2 * np.pi
                target_rpy[i] = self.rad2deg(target_rpy[i])
        print("target_rpy:", [*target_pose, *target_rpy])
        return [*target_pose, *target_rpy]
    
    ## 5. 机器人运动函数
    # 机器人关节点动调节
    def MoveRelJ(self, Axis, Direction, Distance=1, Override=None):
        """
        关节相对运动，相对运动不能连续使用，需要等待上一次运动完成后才能下发下一次运动，否则会报错
        :param Axis: 轴号，0~5 -> J1~J6
        :param Direction: 方向，1：正向；0：负向
        :param Distance: 距离，单位：(°)
        """
        self.SetOverride(Override)
        self.cps.HRIF_MoveRelJ(self.boxID, self.rbtID, Axis, Direction, Distance)
        while not self.IsMotionDone():
            pass
        print(f"Move Axis {Axis} {Distance}° {'Positive' if Direction else 'Negative'}.")
    
    # 机器人直线点动调节
    def MoveRelL(self, Axis, Direction, Distance=1, ToolMotion=1, Override=None):
        """
        直线相对运动，相对运动不能连续使用，需要等待上一次运动完成后才能下发下一次运动，否则会报错
        :param Axis: 轴号，0~5 -> J1~J6
        :param Direction: 方向，1：正向；0：负向
        :param Distance: 距离，单位：(mm)
        :param ToolMotion: 运动坐标类型，1：按TCP坐标运动；0：用户坐标系运动
        """
        self.SetOverride(Override)
        self.cps.HRIF_MoveRelL(self.boxID, self.rbtID, Axis, Direction, Distance, ToolMotion)
        while not self.IsMotionDone():
            pass
        print(f"Move Axis {Axis} {Distance}mm {'Positive' if Direction else 'Negative'}.")
    
    # 机器人路点点动调节
    def MoveRelP(self, Axis, Direction, Distance=1, ToolMotion=1, Override=None):
        """
        路点相对运动，相对运动不能连续使用，需要等待上一次运动完成后才能下发下一次运动，否则会报错
        :param Axis: 轴号，0~5 -> J1~J6
        :param Direction: 方向，1：正向；0：负向
        :param Distance: 距离，单位：(mm)
        :param ToolMotion: 运动坐标类型，1：按TCP坐标运动；0：用户坐标系运动
        """
        self.SetOverride(Override)
        self.cps.HRIF_MoveRelP(self.boxID, self.rbtID, Axis, Direction, Distance, ToolMotion)
        while not self.IsMotionDone():
            pass
        print(f"Move Axis {Axis} {Distance}mm {'Positive' if Direction else 'Negative'}.")

    # 关节运动
    def MoveJ(self, RawACSpoints, Points=[450, 1, 450, -180, 5, 180], TcpName=None, UcsName="Base", Override=None, Radius=50, IsUseJoint=1, IsSeek=0, IOBit=0, IOState=0, CmdID="0"):
        """
        关节运动：
        :param RawACSpoints: 关节目标位置(dJ1 ~ dJ6)，单位：(°)，nIsUseJoint=1时为目标关节坐标，nIsUseJoint=0时仅作参考
        :param Points: 空间目标位置(X, Y, Z, RX, RY, RZ)，单位：(mm, °)，nIsUseJoint=1时无效
        :param TcpName: 工具坐标变量，默认为“TCP”，nIsUseJoint=1时无效
        :param UcsName: 用户坐标变量，默认为“Base”，nIsUseJoint=1时无效
        :param Override: 速度比例，范围：0~1，默认为None
        :param Radius: 过渡半径，单位：(mm)，默认为50
        :param IsUseJoint: 是否使用关节角度，默认为1 -> 0：不使用关节角度；1：使用关节角度
        PS:以下参数为IO检测参数，仅在存在外接电箱IO设备时使用
        :param IsSeek: 是否使用检测DI停止，nIsSeek为1，则开启检测DI停止，如果电箱的nIOBit位索引的DI的状态=nIOState时，机器人停止运动，否则运动到目标点完成运动
        :param IOBit: 检测的DI索引，nIsSeek=0时无效
        :param IOState: 检测的DI状态，nIsSeek=0时无效
        :param CmdID: 路点ID，当前路点ID，可以自定义，也可以按顺序设置为"1"、"2"、"3"
        """
        # 选择TCP坐标
        if TcpName is None:
            TcpName = self.TcpName
        # 计算运行速度
        self.SetOverride(Override=Override)
        # 执行关节运动
        self.cps.HRIF_MoveJ(self.boxID, self.rbtID, Points, RawACSpoints, TcpName, UcsName, self.joint_vel, self.joint_acc, Radius, IsUseJoint, IsSeek, IOBit, IOState, CmdID)
        print(f"Move to {RawACSpoints}.")
    
    # 直线轨迹运动
    def MoveL(self, Points, RawACSpoints=[0, 0, 90, 0, 90, 0], TcpName=None, UcsName="Base", Override=None, Radius=50, IsSeek=0, IOBit=0, IOState=0, CmdID="0"):
        """
        直线轨迹运动：
        :param Points: 空间目标位置(X, Y, Z, RX, RY, RZ)，单位：(mm, °)
        :param RawACSpoints: 关节目标位置(dJ1 ~ dJ6)，单位：(°)，仅作参考
        :param TcpName: 工具坐标变量，默认为“TCP”
        :param UcsName: 用户坐标变量，默认为“Base”
        :param Override: 速度比例，范围：0~1，默认为None
        :param Radius: 过渡半径，单位：(mm)，默认为50
        PS:以下参数为IO检测参数，仅在存在外接电箱IO设备时使用
        :param IsSeek: 是否使用检测DI停止，nIsSeek为1，则开启检测DI停止，如果电箱的nIOBit位索引的DI的状态=nIOState时，机器人停止运动，否则运动到目标点完成运动
        :param IOBit: 检测的DI索引，nIsSeek=0时无效
        :param IOState: 检测的DI状态，nIsSeek=0时无效
        :param CmdID: 路点ID，当前路点ID，可以自定义，也可以按顺序设置为"1"、"2"、"3"
        """
        # 选择TCP坐标
        if TcpName is None:
            TcpName = self.TcpName
        # 计算运行速度
        self.SetOverride(Override=Override)
        # 执行关节运动
        self.cps.HRIF_MoveL(self.boxID, self.rbtID, Points, RawACSpoints, TcpName , UcsName, self.linear_vel, self.linear_acc, Radius, IsSeek, IOBit, IOState, CmdID)
        print(f"Move to {Points}.")
    
    # 圆弧轨迹运动
    def MoveC(self, StartPoint, AuxPoint, EndPoint, FixedPosure=0, MoveCType=0, RadLen=1, Override=None, Radius=50, TcpName=None, UcsName="Base", CmdID="0"):
        """
        圆弧轨迹运动：
        :param StartPoint: 圆弧起始点位置(X, Y, Z, RX, RY, RZ)，单位：(mm, °)
        :param AuxPoint: 圆弧辅助点位置(X, Y, Z, RX, RY, RZ)，单位：(mm, °)
        :param EndPoint: 圆弧结束点位置(X, Y, Z, RX, RY, RZ)，单位：(mm, °)，如果用整圆，结束点也是圆上的一个经过点，整圆跑完后才停止
        :param FixedPosure: 圆弧整个运动过程中是否保持姿态不变，默认为0 -> 0：固定姿态；1：不固定姿态
        :param MoveCType: 圆弧类型，默认为0 -> 0：整圆；1：圆弧
        :param RadLen: 弧长（圆周数），使用整圆运动时表示整圆的圈数，小数部分无效；当使用圆弧运动时无效，仅通过三个点位确定圆弧路径
        :param Radius: 过渡半径，单位：(mm)，默认为50
        :param TcpName: 工具坐标变量
        :param UcsName: 用户坐标变量
        :param trCmdID: 路点ID，当前路点ID，可以自定义，也可以按顺序设置为"1"、"2"、"3"
        """
        # 选择TCP坐标
        if TcpName is None:
            TcpName = self.TcpName
        # 计算运行速度
        self.SetOverride(Override=Override)
        # 执行关节运动
        self.cps.HRIF_MoveC(self.boxID, self.rbtID, StartPoint , AuxPoint, EndPoint, FixedPosure, MoveCType, RadLen, self.linear_vel, self.linear_acc, Radius, TcpName , UcsName, CmdID)
        print(f"Move to {EndPoint}.")
    
    # 机器人高精度移动函数
    def HP_Move(self, target_pose, Move_Type="L", Override=None, *args):
        """
        机器人高精度（High-Precision）移动函数：
        该函数运行时，会检测机器人是否到达目标位置，如果没有到达目标位置，会再次执行移动操作，直到到达目标位置或连续移动三次后退出
        该函数运行后，无法Stop或Interrupt，谨慎使用！
        :param target_pose: 目标位置
        :param Move_Type: 移动类型，"J"：关节坐标，"L"：直线坐标，"C"：圆弧坐标（list: 包含三个坐标）
        :param Velocity: 速度，单位：(mm/s, °/s)
        :param Acc: 加速度，单位：(mm/s^2, °/s^2)
        :param args: Move函数包含的其他参数
        """
        index = 0
        # 关节坐标移动
        if Move_Type == "J":
            self.MoveJ(RawACSpoints=target_pose, Override=Override)
            while not self.IsMotionDone():
                pass
            while not self.IsPoseClose(target_pose) and index < 3: # 满足精度条件或连续移动三次后退出
                self.MoveJ(RawACSpoints=target_pose, Override=Override)
                index += 1
                print(f"Perform the {index}th additional move")
        # 直线坐标移动
        elif Move_Type == "L":
            self.MoveL(Points=target_pose, Override=Override)
            while not self.IsMotionDone():
                pass
            while not self.IsPoseClose(target_pose) and index < 3: # 满足精度条件或连续移动三次后退出
                self.MoveL(Points=target_pose, Override=Override)
                index += 1
                print(f"Perform the {index}th additional move")
        # 圆弧坐标移动
        elif Move_Type == "C":
            self.MoveC(StartPoint=target_pose[0], AuxPoint=target_pose[1], EndPoint=target_pose[2], Override=Override)
            while not self.IsMotionDone():
                pass
            while not self.IsPoseClose(target_pose[2]) and index < 3: # 满足精度条件或连续移动三次后退出
                self.MoveC(StartPoint=target_pose[0], AuxPoint=target_pose[1], EndPoint=target_pose[2], Override=Override)
                index += 1
                print(f"Perform the {index}th additional move")
        else:
            print("Move_Type Error!")

    ## 6. 机器人抓取函数
    # 机器人平面抓取物体
    def Grasp_plane(self, target_pose, target_yaw=None, Override=None):
        """
        机器人抓取物体后不动：
        :param target_pose: 目标位置，[X, Y, Z]
        :param target_rpy: 机器人抓取角度，[RX, RY, RZ]
        """
        # 计算RPY角度
        if target_yaw is not None:
            target_rpy = self.target_rpy
            target_rpy[2] += target_yaw
        else:
            target_rpy = None
        # 判断目标点是否在工作空间内
        target_pose = self.IsInWorkspace(target_pose, target_rpy)
        if target_pose == False:
            print("The target pose is not in the workspace.")
            return
        # 机器人到达抓取零点
        # self.MoveJ(RawACSpoints=self.home_joint_config, Override=Override)
        # 打开夹爪
        self.OpenEndEffector()
        # 机器人到达抓取预备位置（抓取点上方50mm）
        grasp_pre_pose = copy.deepcopy(target_pose)
        grasp_pre_pose[2] += 50
        self.MoveL(Points=grasp_pre_pose, Override=Override)
        while not self.IsMotionDone():
            pass
        time.sleep(1)
        # 机器人到达抓取位置
        self.MoveL(Points=target_pose, Override=Override*0.3)
        while not self.IsMotionDone():
            pass
        # 闭合夹爪
        self.CloseEndEffector()
        time.sleep(1)
        # 机器人到达抓取预备位置（抓取点上方50mm）
        self.MoveL(Points=grasp_pre_pose, Override=Override)
        print("Finish the grasp and place test.")

    # 机器人抓取放置物体
    def Grasp(self, target_pose, target_yaw=None, Override=None):
        """
        机器人抓取物体后将其放在指定位置：
        :param target_pose: 目标位置，[X, Y, Z]
        :param target_rpy: 机器人抓取角度，[RX, RY, RZ]
        """
        # 计算RPY角度
        if target_yaw is not None:
            target_rpy = self.target_rpy
            target_rpy[2] += target_yaw
        else:
            target_rpy = None
        # 判断目标点是否在工作空间内
        target_pose = self.IsInWorkspace(target_pose, target_rpy)
        if target_pose == False:
            print("The target pose is not in the workspace.")
            return
        # # 机器人到达抓取零点
        # self.MoveJ(RawACSpoints=self.home_joint_config, Override=Override)
        # 打开夹爪
        self.OpenEndEffector()
        # 机器人到达抓取预备位置（抓取点上方50mm）
        grasp_pre_pose = copy.deepcopy(target_pose)
        grasp_pre_pose[2] += 120
        self.MoveL(Points=grasp_pre_pose, Override=Override)
        while not self.IsMotionDone():
            pass
        # 机器人到达抓取位置
        self.MoveL(Points=target_pose, Override=Override*0.3)
        while not self.IsMotionDone():
            pass
        # 闭合夹爪
        self.CloseEndEffector()
        # 机器人到达抓取预备位置（抓取点上方50mm）
        self.MoveL(Points=grasp_pre_pose, Override=Override)
        while not self.IsMotionDone():
            pass
        # 机器人到达抓取零点
        self.MoveJ(RawACSpoints=self.home_joint_config, Override=Override)
        while not self.IsMotionDone():
            pass
        # 机器人到达放置预备位置（放置点上方50mm）
        # place_pre_pose = copy.deepcopy(self.place_position)
        place_pre_pose = copy.deepcopy(target_pose)
        place_pre_pose[2] += 120
        self.MoveL(Points=place_pre_pose, Override=Override)
        while not self.IsMotionDone():
            pass
        # 机器人到达放置位置
        place_pose = copy.deepcopy(target_pose)
        place_pose[2] += 70
        self.MoveL(Points=place_pose, Override=Override*0.3)
        while not self.IsMotionDone():
            pass
        # 打开夹爪
        self.OpenEndEffector()
        # 机器人到达预备放置位置（放置点上方50mm）
        self.MoveL(Points=place_pre_pose, Override=Override)
        # # 机器人到达放置零点
        # self.MoveJ(RawACSpoints=self.home_joint_config, Override=Override)
        print("Finish the grasp and place test.")

    # 机器人推动物体
    def Push(self, target_pose, target_yaw=None, move_orientation=0, length=0.1, Override=None):
        """
        机器人推动物体：
        :param target_pose: 目标位置，[X, Y, Z]
        :param target_rpy: 机器人抓取角度，[RX, RY, RZ]
        :param move_orientation: 推动方向，单位：(°)
        :param length: 推动长度，单位：(mm)
        :param Override: 速度比例，范围：0~1，默认为None
        """
        # 计算RPY角度
        if target_yaw is not None:
            target_rpy = self.target_rpy
            target_rpy[2] += target_yaw
        else:
            target_rpy = None
        # 判断目标点是否在工作空间内
        target_pose = self.IsInWorkspace(target_pose, target_rpy)
        if target_pose == False:
            print("The target pose is not in the workspace.")
            return
        # 计算推动位置
        for i in range(2):
            target_pose[i] = min(max(target_pose[i], self.workspace_limits[0][i]+50), self.workspace_limits[1][i]-50)
        target_pose[2] = min(max(target_pose[2], self.workspace_limits[0][2]), self.workspace_limits[1][2])
        print('Executing: push at (%f, %f, %f) and the orientation is %f' % (target_pose[0], target_pose[1], target_pose[2], move_orientation))

        # 机器人到达推动零点
        self.MoveJ(Points=self.home_joint_config, Override=Override)
         # 闭合夹爪
        self.CloseEndEffector()
        # 机器人到达推动预备位置（推动点上方50mm）
        push_pre_pose = copy.deepcopy(target_pose)
        push_pre_pose[2] += 50
        self.MoveL(Points=push_pre_pose, Override=Override)
        # 机器人到达推动位置
        self.MoveL(Points=target_pose, Override=Override)
        # 计算推动目标位置
        push_target_pose = copy.deepcopy(target_pose)
        push_target_pose[0] += length * np.cos(move_orientation)
        push_target_pose[1] += length * np.sin(move_orientation)
        # 机器人到达推动目标位置
        self.MoveL(Points=push_target_pose, Override=Override)
        # 机器人到达推动预备位置（推动点上方50mm）
        self.MoveL(Points=push_pre_pose, Override=Override)
        # 机器人到达推动零点
        self.MoveJ(Points=self.home_joint_config, Override=Override)


if __name__ == "__main__":
    # 机器人初始化
    robot = HansRobot()
    print("Robot is ready!")
        
    
    
    
    
    
    



    

    

        







