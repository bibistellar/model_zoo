# -*- coding: utf-8 -*-
# @Organization  : ncse.ai
# @Function      : GenderAge Ray Actor 封装

import cv2
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv
import os
import onnxruntime as ort
import sys
import os
from skimage import transform as trans

# 添加上级目录到路径，以便可以导入utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

ort.set_default_logger_severity(3)

def transform(data, center, output_size, scale, rotation):
    """
    对图像进行一系列变换，包括缩放、平移和旋转
    
    参数:
        data: 输入图像
        center: 变换中心点坐标 [x, y]
        output_size: 输出图像的尺寸
        scale: 缩放比例
        rotation: 旋转角度（度）
    
    返回:
        cropped: 变换后的图像
        M: 变换矩阵
    """
    scale_ratio = scale
    # 将角度转换为弧度
    rot = float(rotation) * np.pi / 180.0
    
    # 执行一系列变换：缩放、平移到原点、旋转、平移到目标中心
    # 1. 缩放
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    
    # 2. 平移到原点
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    
    # 3. 旋转
    t3 = trans.SimilarityTransform(rotation=rot)
    
    # 4. 平移到输出图像中心
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    
    # 组合所有变换
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    
    # 应用仿射变换
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M

# 初始化Ray - 如果在本地运行，可以直接初始化
# 如果已经有集群，则连接到集群
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

@ray.remote(num_cpus=1, num_gpus=0, runtime_env=conda_env)
class GenderAgeDetector:
    """Ray Actor 封装GenderAge检测功能"""
    
    def __init__(self, model_file=None, tpu_id=-1):
        """初始化GenderAge检测器"""
        # 默认查找当前目录下的模型文件
        if model_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_file = os.path.join(current_dir, "genderage.onnx")
            if not os.path.exists(model_file):
                print(f"警告: 默认模型文件不存在: {model_file}")
                print("请提供有效的模型文件路径")
                return
        
        # 初始化属性
        self.model_file = model_file
        self.taskname = 'genderage'
        if tpu_id < 0:
            tpu_id = 0
        
        self.input_mean = 0.0
        self.input_std = 1.0
        
        # 初始化模型会话
        try:
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(
                self.model_file,
                providers=providers
            )
            
            # 获取会话元数据
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.input_shape = inp.shape  # [N, C, H, W]
            
            # 计算输入大小 (W, H)
            if isinstance(self.input_shape[-2], int) and isinstance(self.input_shape[-1], int):
                self.input_size = (
                    self.input_shape[-1], self.input_shape[-2]
                )
            else:
                self.input_size = None
                print("警告: 无法确定模型输入尺寸")
            
            outputs = self.session.get_outputs()
            assert len(outputs) == 1, "GenderAge模型应该只有一个输出"
            self.output_name = outputs[0].name
            self.output_shape = outputs[0].shape  # 例如 [1,3]
            
            print(f"已加载GenderAge模型: {model_file}")
            print(f"输入尺寸: {self.input_size}")
        except Exception as e:
            print(f"加载模型时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get(self, img, bbox):
        """
        获取人脸性别和年龄
        
        Args:
            img: 输入图像(BGR格式)
            bbox: 人脸边界框 (x1, y1, x2, y2)
            
        Returns:
            gender: 性别 (0: 女性, 1: 男性)
            age: 年龄
        """
        if img is None:
            print("警告: 输入图像为None")
            return None, None
            
        try:
            # 从边界框提取数据
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            scale = self.input_size[0] / (max(w, h) * 1.5)
            
            # 进行人脸对齐
            aligned, _ =  transform(
                img, center, self.input_size[0], scale, rotation=0
            )

            # 创建blob
            blob = cv2.dnn.blobFromImage(
                aligned,
                1.0 / self.input_std,
                self.input_size,
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=True,
            ).astype(np.float32)
            
            # 前向推理
            raw_out = self.session.run([self.output_name], {self.input_name: blob})[0][0]
            # raw_out: [gender_prob0, gender_prob1, age_norm]
            
            # 解析结果
            gender = int(np.argmax(raw_out[:2]))
            age = int(np.round(raw_out[2] * 100))
            
            return gender, age
        except Exception as e:
            print(f"性别年龄检测时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None