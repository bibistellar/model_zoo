import platform
import sys
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import ray
from ray.runtime_env import RuntimeEnv
import os
from insightface.app import FaceAnalysis
import onnxruntime

from runtime_config import conda_env

# 提供人脸对齐功能
def norm_crop(img, landmark, image_size=112, mode='arcface'):
    """
    根据人脸关键点对图像进行标准化裁剪
    
    参数:
        img: 输入图像
        landmark: 人脸关键点坐标，形状为(5, 2)
        image_size: 输出图像的尺寸
        mode: 对齐模式
        
    返回:
        warped: 对齐后的人脸图像
    """
    from skimage import transform as trans
    
    # 这些是ArcFace使用的标准关键点坐标
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)
    
    # 将目标坐标调整为正确大小
    if image_size != 112:
        src = src * (image_size / 112.0)
    
    # 估计变换
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, src)
    M = tform.params[0:2, :]
    
    # 应用变换
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

# 初始化Ray - 如果在本地运行，可以直接初始化
# 如果已经有集群，则连接到集群
@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class ArcFaceRecognition:
    """Ray Actor 封装人脸特征提取功能"""
    
    def __init__(self):
        """初始化ArcFace模型"""
        # 默认使用人脸检测
        self.providers = ['CPUExecutionProvider']
        self.face_analysis = FaceAnalysis(providers=self.providers)
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        
        # 标准参数，后续会根据模型进行调整
        self.input_size = (112, 112)
        self.input_mean = 127.5
        self.input_std = 127.5
        
        # 默认加载当前目录下的模型
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_model = os.path.join(current_dir, "w600k_r50.onnx")
        if os.path.exists(default_model):
            self.load_model(default_model)
        else:
            print(f"默认模型不存在: {default_model}")
            self.session = None
    
    def load_model(self, model_file=None, use_gpu=False, ismxnet=False):
        """
        加载ONNX ArcFace模型
        
        Args:
            model_file: 模型文件路径
            use_gpu: 是否使用GPU
            ismxnet: 是否是MXNet模型导出的，影响输入处理方式
            
        Returns:
            成功加载返回True，否则返回False
        """
        try:
            if model_file is None:
                print("未提供模型文件路径")
                return False
                
            # 设置ONNX Runtime提供程序
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            # 加载ONNX模型
            self.session = onnxruntime.InferenceSession(model_file, providers=providers)
            
            # 获取输入信息
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            self.input_shape = inp.shape  # [N, C, H, W]
            if isinstance(self.input_shape[-2], int) and isinstance(self.input_shape[-1], int):
                self.input_size = (self.input_shape[-1], self.input_shape[-2])
            else:
                self.input_size = (112, 112)  # 默认尺寸
            
            # 获取输出信息
            outputs = self.session.get_outputs()
            self.output_name = outputs[0].name
            self.output_shape = outputs[0].shape
            
            # 根据模型类型设置归一化参数
            if ismxnet:
                # MXNet模型
                self.input_mean = 0.0
                self.input_std = 1.0
            else:
                # 其他模型
                self.input_mean = 127.5
                self.input_std = 127.5
                
            print(f"成功加载模型: {model_file}")
            print(f"输入形状: {self.input_shape}, 输入名称: {self.input_name}")
            print(f"输出形状: {self.output_shape}, 输出名称: {self.output_name}")
            print(f"输入归一化参数: 均值={self.input_mean}, 标准差={self.input_std}")
            
            return True
        except Exception as e:
            print(f"加载模型时发生错误: {str(e)}")
            return False

    def get_face_embedding(self, image, landmark=None):
        """
        提取人脸特征向量
        
        Args:
            image: 输入图像 (BGR格式)
            landmark: 可选，人脸关键点。如果提供，则使用这些关键点进行对齐
                      否则使用insightface库的FaceAnalysis类中提供的人脸检测方法来获取人脸
            face_idx: 如果检测到多个人脸，指定使用第几个人脸作为基准(默认使用第一个)
            
        Returns:
            embedding: 人脸特征向量
            aligned_face: 对齐后的人脸图像
        """
        face_idx = 0
        if image is None:
            print("警告: 输入图像为None")
            return None, None
            
        if self.session is None:
            print("错误: 模型未加载")
            return None, None
            
        try:
            # 如果没有提供landmark，则检测人脸
            if landmark is None:
                img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                faces = self.face_analysis.get(img_rgb)
                if len(faces) == 0:
                    print("未检测到人脸")
                    return None, None
                    
                # 获取指定索引的人脸
                if face_idx >= len(faces):
                    face_idx = 0  # 如果索引超出范围，使用第一个人脸
                face = faces[face_idx]
                
                # 获取关键点并进行对齐
                if hasattr(face, 'kps'):
                    landmark = face.kps
                    aligned_face = norm_crop(image, landmark)
                else:
                    print("未找到人脸关键点")
                    return None, None
                
                # 绘制人脸边界框和关键点
                result_img = image.copy()
                bbox = face.bbox.astype(int)
                cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 绘制关键点
                kps = landmark.astype(np.int32)
                for i in range(5):
                    cv2.circle(result_img, (kps[i][0], kps[i][1]), 1, (0, 0, 255), 2)
            else:
                # 使用提供的landmark进行人脸对齐
                aligned_face = norm_crop(image, landmark)
                result_img = aligned_face.copy()
                
            # 使用ONNX模型提取特征
            # 预处理图像
            blob = cv2.dnn.blobFromImage(aligned_face, 1.0/self.input_std, self.input_size,
                                        (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
            
            # 进行推理
            embedding = self.session.run([self.output_name], {self.input_name: blob})[0]
            # 转换为一维向量
            embedding = embedding.flatten()
            
            return embedding
                
        except Exception as e:
            print(f"提取人脸特征时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
            
    def compute_similarity(self, feat1, feat2):
        """
        计算两个人脸特征向量的相似度
        
        Args:
            feat1: 第一个人脸特征向量
            feat2: 第二个人脸特征向量
            
        Returns:
            similarity: 相似度得分，范围[-1, 1]，越接近1表示越相似
        """
        if feat1 is None or feat2 is None:
            return -1.0
            
        try:
            # 确保使用一维向量
            feat1 = feat1.ravel()
            feat2 = feat2.ravel()
            
            # 计算余弦相似度
            similarity = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
            return float(similarity)
        except Exception as e:
            print(f"计算相似度时发生错误: {str(e)}")
            return -1.0
            
    def compare_faces(self, image1, image2):
        """
        比较两张图像中的人脸相似度
        
        Args:
            image1: 第一张图像 (BGR格式)
            image2: 第二张图像 (BGR格式)
            
        Returns:
            similarity: 相似度得分
            result_image: 可视化结果
        """
        if image1 is None or image2 is None:
            print("警告: 输入图像为None")
            return -1.0, None
            
        try:
            # 获取第一张图像的人脸特征
            embedding1 = self.get_face_embedding(image1)
            if embedding1 is None:
                return -1.0
                
            # 获取第二张图像的人脸特征
            embedding2 = self.get_face_embedding(image2)
            if embedding2 is None:
                return -1.0
                
            # 计算相似度
            similarity = self.compute_similarity(embedding1, embedding2)
            
            # 添加相似度文本
            print(f"相似度: {similarity:.4f}")
                        
            return similarity
            
        except Exception as e:
            print(f"比较人脸时发生错误: {str(e)}")
            return -1.0, None


