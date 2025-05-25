import numpy as np
from numpy.linalg import norm
import cv2
from cv2.typing import MatLike
import ray

import runtime_config as runtime_config

from models.buffalo_l import ray_buffalo_l_onnx

class FaceModel:
    def init(self):
            #检测是否已连接ray集群，没有则连接，有则跳过
            ray.init(address=runtime_config.ray_address,ignore_reinit_error=True)  # 连接到集群
    def getfaces(self, img:MatLike) -> tuple:
        """
        获取图像中的人脸区域,使用buffalo_l模型（det_10g.onnx）进行人脸区域的获取
        Args:
            img: 输入图像（BGR格式）
        Returns:
            faces: 人脸区域列表，每个元素为 (x1, y1, x2, y2, score)
        """
        if img is None:
            print("警告: 输入图像为None")
            return []
        
        try:
            # 创建Buffalo L模型实例
            buffalo_l = ray.get_actor(runtime_config.buffalo_l_config["name"], namespace=runtime_config.buffalo_l_config["namespace"])
            dets_ref = ray.get(buffalo_l.detect.remote(img=img, max_num=0))  # 这样不会报错
            # 检测人脸
            dets, kprets = dets_ref

            return dets, kprets
        except Exception as e:
            print(f"人脸检测失败: {str(e)}")
            return [], None
    def getfeat_norm(self, img, landmark):
        """
        使用Arcface模型（w600k_r50）获取人脸区域的特征向量为对比做准备
           Args:
               img: 输入图像（BGR格式）
               landmark: 人脸关键点
           Returns:
               feat_norm: 归一化的人脸特征向量
        """
        if img is None or landmark is None:
            print("警告: 输入图像或人脸关键点为None")
            return None
        
        try:
            # 创建Arcface模型实例
            arcface = ray.get_actor(runtime_config.arcface_config["name"], namespace=runtime_config.arcface_config["namespace"])
            feat_norm = ray.get(arcface.get_face_embedding.remote(image=img, landmark=landmark))
            return feat_norm
        except Exception as e:
            print(f"获取人脸特征失败: {str(e)}")
            return None