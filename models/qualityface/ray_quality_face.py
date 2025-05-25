from pdb import run
import platform
import sys
import cv2
import numpy as np
import qualityface
import matplotlib.pyplot as plt
import ray
from ray.runtime_env import RuntimeEnv
import logging
import os

# ray_address = "ray://localhost:10001" # Ray集群地址

# 初始化Ray - 如果在本地运行，可以直接初始化
# 如果已经有集群，则连接到集群
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)
@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class BuffaloFaceDetector:
    """Ray Actor 封装人脸检测功能"""
    
    def __init__(self):
        """初始化F"""

    def print_sysytem_info(self):
        """打印系统信息"""
        print(f"Python版本: {sys.version}")
        print(f"Python版本详情: {platform.python_version()}")
        print(f"Python实现: {platform.python_implementation()}")
        # 打印当前的pip包
        print(f"当前pip包: {os.popen('pip freeze').read()}")
        # 打印当前conda环境
        print(f"当前conda环境: {os.popen('conda env list').read()}")
        return {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "pip_packages": os.popen('pip freeze').read(),
            "conda_env": os.popen('conda env list').read()
        }

    def detect_face(self,image):
        """
        检测图片中的人脸
        
        Args:
            image: 图像数据（BGR格式）或图像路径
            
        Returns:
            faces: 检测到的人脸
        """
        #判断输入是否为None
        if image is None:
            print("警告: 输入图像为None")
            return []
        
        # 创建图像的可写副本
        img = image.copy()
        
        try:
            # 转换为RGB格式
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 检测人脸
            faces = self.app.get(img_rgb)
            # 打印检测到的人脸数量
            for face in faces:
                # Get the bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw facial landmarks
                if hasattr(face, 'landmark_2d_106'):
                    landmark = face.landmark_2d_106.astype(np.int32)
                    for pt in landmark:
                        cv2.circle(img, (pt[0], pt[1]), 1, (0, 0, 255), -1)

                # Add confidence score text
                confidence = face.det_score
                text = f"Conf: {confidence:.2f}"
                cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return img
        except Exception as e:
            print(f"检测人脸时发生错误: {str(e)}")
            return []



# 连接到本地Ray集群
# 注意：请确保先运行了main.py启动本地集群
ray.init(address="auto",ignore_reinit_error=True) # 连接到本地集群，无需指定地址
try:
     actor = ray.get_actor("Buffalo", namespace="face_detection")
     ray.kill(actor)
     print(f"已终止Actor: Buffalo")
except:
     print(f"Actor Buffalo")
deploy = BuffaloFaceDetector.options(
    name="Buffalo",
    namespace="face_detection",
    lifetime="detached",
    max_restarts=-1,
    num_cpus=1,
    num_gpus=1
).remote()
print("Buffalo已部署")
# status = ray.get(deploy.print_sysytem_info.remote())