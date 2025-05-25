# filepath: /root/Code/aiplat/model_zoo/models/openpose/ray_openpose.py
import platform
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import ray
from ray.runtime_env import RuntimeEnv
import logging
import os
import time
import math
from scipy.ndimage.filters import gaussian_filter

# 导入本地模块
import util
from body import Body
from model import bodypose_model

# 初始化Ray - 如果在本地运行，可以直接初始化
# 如果已经有集群，则连接到集群
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class OpenPoseDetector:
    """Ray Actor 封装人体姿态检测功能"""
    
    def __init__(self, model_path='body_pose_model.pth', use_cuda=True):
        """初始化OpenPose应用"""
        self.use_cuda = use_cuda and torch.cuda.is_available()
        # 创建模型实例
        self.model = Body(model_path, cuda=self.use_cuda)
        
        # 打印GPU状态信息
        if self.use_cuda:
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU进行推理")
    
    def print_system_info(self):
        """打印系统信息"""
        print(f"Python版本: {sys.version}")
        print(f"Python版本详情: {platform.python_version()}")
        print(f"Python实现: {platform.python_implementation()}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        
        # 打印当前的pip包
        print(f"当前pip包: {os.popen('pip freeze').read()}")
        # 打印当前conda环境
        print(f"当前conda环境: {os.popen('conda env list').read()}")
        
        return {
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "pip_packages": os.popen('pip freeze').read(),
            "conda_env": os.popen('conda env list').read()
        }

    def detect_pose(self, image):
        """
        检测图片中的人体姿态
        
        Args:
            image: 图像数据（BGR格式）或图像路径
            
        Returns:
            processed_image: 处理后的图像
            keypoints: 检测到的关键点
        """
        # 判断输入是否为None
        if image is None:
            print("警告: 输入图像为None")
            return None, []
        
        try:
            # 判断输入是路径还是图像
            if isinstance(image, str):
                # 如果是路径，则读取图像
                img = cv2.imread(image)
                if img is None:
                    print(f"警告: 无法读取图像文件: {image}")
                    return None, []
            else:
                # 如果是图像数据，则创建副本
                img = image.copy()
            
            # 确保图像不为None并且有正确的形状
            if img is None or img.size == 0:
                print("警告: 图像为空或无效")
                return None, []
            
            # 打印设备信息
            if self.use_cuda and torch.cuda.is_available():
                print(f"CUDA可用，使用设备: {torch.cuda.get_device_name(0)}")
            else:
                print("使用CPU进行推理")
                
            start_time = time.time()
            # 检测人体姿态
            keypoints = self.model(img)
            end_time = time.time()
            
            print(f"姿态检测完成，用时: {end_time - start_time:.4f} 秒")
            print(f"检测到 {len(keypoints)} 个人体")
            
            # 绘制姿态关键点
            canvas = util.draw_bodypose18(img, keypoints)
            
            return canvas, keypoints
            
        except Exception as e:
            print(f"检测人体姿态时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, []

    def check_gpu_usage(self):
        """检查GPU使用情况并返回详细信息"""
        gpu_info = {}
        
        # 检查CUDA可用性
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["current_device"] = torch.cuda.current_device()
            # 获取GPU内存使用情况
            gpu_info["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
            gpu_info["memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB"
        
        # 使用nvidia-smi获取更详细信息
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi']).decode('utf-8')
            gpu_info["nvidia_smi"] = result
        except Exception as e:
            gpu_info["nvidia_smi_error"] = str(e)
        
        return gpu_info


# 连接到Ray集群
if __name__ == "__main__":
    ray.init(address="auto", ignore_reinit_error=True)  # 连接到本地集群
    
    try:
        # 尝试获取已有Actor
        actor = ray.get_actor("OpenPose", namespace="pose_detection")
        ray.kill(actor)
        print(f"已终止Actor: OpenPose")
    except:
        print(f"Actor OpenPose不存在或已停止")
    
    # 部署新的Actor
    deploy = OpenPoseDetector.options(
        name="OpenPose",
        namespace="pose_detection",
        lifetime="detached",
        max_restarts=-1,
        num_cpus=1,
        num_gpus=0
    ).remote()
    
    print("OpenPose已部署")
    
    # # 打印系统信息
    # ray.get(deploy.print_system_info.remote())
    
    # # 检查GPU使用情况
    # gpu_info = ray.get(deploy.check_gpu_usage.remote())
    # print("\nGPU使用情况:")
    # for key, value in gpu_info.items():
    #     if key != "nvidia_smi":  # nvidia_smi输出太长，这里不打印
    #         print(f"  {key}: {value}")