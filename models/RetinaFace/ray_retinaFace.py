
import platform
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.runtime_env import RuntimeEnv
import logging
import os
from retinaface import RetinaFace

# 初始化Ray - 如果在本地运行，可以直接初始化
# 如果已经有集群，则连接到集群
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class RetinaFaceDetector:
    """Ray Actor 封装RetinaFace人脸检测功能"""
    
    def __init__(self):
        """初始化RetinaFace检测器"""
        self.detector = RetinaFace
        # 可以根据需要设置检测阈值
        self.det_threshold = 0.5
        
    def print_system_info(self):
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

    def detect_face(self, image_path=None, image=None):
        """
        检测图片中的人脸
        
        Args:
            image_path: 图像路径
            image: 图像数据（BGR格式）
            
        Returns:
            faces: 检测到的人脸信息和标注后的图像
        """
        # 判断输入
        if image_path is None and image is None:
            print("警告: 没有提供输入图像")
            return None, {}
        
        try:
            # 如果提供了图像路径，则从文件加载图像
            if image_path is not None:
                img = cv2.imread(image_path)
            else:
                # 创建图像的可写副本
                img = image.copy()
            
            # 检测人脸
            faces = self.detector.detect_faces(img if image_path is None else image_path)
            
            # 检查是否检测到人脸
            if not faces:
                print("未检测到人脸")
                return img, {}
            
            # 创建标注后的图像副本
            annotated_img = img.copy()
            
            # 在图像上标注人脸
            for face_id, face_info in faces.items():
                # 获取边界框
                bbox = face_info['facial_area']
                x1, y1, x2, y2 = bbox
                
                # 绘制边界框
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制关键点
                landmarks = face_info['landmarks']
                for point_name, (x, y) in landmarks.items():
                    # 根据点的类型使用不同颜色
                    if 'eye' in point_name:
                        color = (255, 0, 0)  # 蓝色
                    elif 'nose' in point_name:
                        color = (0, 255, 0)  # 绿色
                    else:  # mouth
                        color = (0, 0, 255)  # 红色
                    
                    cv2.circle(annotated_img, (int(x), int(y)), 2, color, -1)
                
                # 添加置信度分数文字
                confidence = face_info['score']
                text = f"Conf: {confidence:.2f}"
                cv2.putText(annotated_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            return annotated_img, faces
            
        except Exception as e:
            print(f"检测人脸时发生错误: {str(e)}")
            return None, {}
    
    def extract_aligned_faces(self, image_path=None, image=None):
        """
        提取并对齐图像中的人脸
        
        Args:
            image_path: 图像路径
            image: 图像数据（BGR格式）
            
        Returns:
            aligned_faces: 对齐后的人脸图像列表
        """
        try:
            # 如果提供了图像路径，则使用路径
            if image_path is not None:
                aligned_faces = self.detector.extract_faces(img_path=image_path, align=True)
            # 否则使用提供的图像数据
            elif image is not None:
                # 先保存图像到临时文件
                temp_path = "temp_image_for_alignment.jpg"
                cv2.imwrite(temp_path, image)
                aligned_faces = self.detector.extract_faces(img_path=temp_path, align=True)
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                print("警告: 没有提供输入图像")
                return []
            
            return aligned_faces
            
        except Exception as e:
            print(f"提取对齐人脸时发生错误: {str(e)}")
            return []


# 连接到Ray集群
ray.init(address="auto", ignore_reinit_error=True)  # 连接到本地集群，无需指定地址

# 尝试停止已存在的Actor
try:
    actor = ray.get_actor("RetinaFace", namespace="face_detection")
    ray.kill(actor)
    print(f"已终止Actor: RetinaFace")
except:
    print(f"未找到运行中的RetinaFace Actor或无法终止")

# 部署RetinaFace Actor
deploy = RetinaFaceDetector.options(
    name="RetinaFace",
    namespace="face_detection",
    lifetime="detached",  # 确保Actor在Ray集群运行期间保持活跃
    max_restarts=-1,      # 无限重启
    num_cpus=1,
    num_gpus=0            # 如果需要GPU加速
).remote()

print("RetinaFace已部署")

# 打印系统信息以验证部署
# status = ray.get(deploy.print_system_info.remote())
# print(status)

# 示例用法
def process_image(image_path):
    """示范如何使用部署的RetinaFace Actor处理图像"""
    result_img, faces = ray.get(deploy.detect_face.remote(image_path=image_path))
    
    if result_img is not None:
        # 显示结果
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"检测到 {len(faces)} 个人脸")
        plt.show()
        
        # 提取对齐的人脸
        aligned_faces = ray.get(deploy.extract_aligned_faces.remote(image_path=image_path))
        
        # 显示对齐后的人脸
        if aligned_faces:
            fig, axes = plt.subplots(1, len(aligned_faces), figsize=(2*len(aligned_faces), 2))
            if len(aligned_faces) == 1:
                axes = [axes]
            for i, face in enumerate(aligned_faces):
                axes[i].imshow(face)
                axes[i].axis('off')
                axes[i].set_title(f"Face {i+1}")
            plt.tight_layout()
            plt.show()
    else:
        print("处理图像失败")

# 示例调用
# process_image("path_to_your_image.jpg")