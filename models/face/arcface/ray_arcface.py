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
from insightface.model_zoo.arcface_onnx import ArcFaceONNX

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
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class ArcFaceRecognition:
    """Ray Actor 封装人脸特征提取功能"""
    
    def __init__(self):
        """初始化ArcFace模型"""
        # 默认使用InsightFace中的ArcFace ONNX模型
        self.providers = ['CPUExecutionProvider']
        self.face_analysis = FaceAnalysis(providers=self.providers)
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))
        
        # 标准参数
        self.input_size = (112, 112)
        self.input_mean = 127.5
        self.input_std = 128.0
    
    def load_model(self, model_file=None, use_gpu=False):
        """
        加载自定义ArcFace模型
        
        Args:
            model_file: 模型文件路径
            use_gpu: 是否使用GPU
            
        Returns:
            成功加载返回True，否则返回False
        """
        try:
            if use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                
            if model_file is not None:
                # 使用InsightFace中的ArcFace ONNX加载器
                self.model = ArcFaceONNX(model_file=model_file, providers=providers)
                print(f"成功加载模型: {model_file}")
            else:
                # 使用默认模型
                pass
                
            return True
        except Exception as e:
            print(f"加载模型时发生错误: {str(e)}")
            return False

    def get_face_embedding(self, image, landmark=None, face_idx=0):
        """
        提取人脸特征向量
        
        Args:
            image: 输入图像 (BGR格式)
            landmark: 可选，人脸关键点。如果提供，则使用这些关键点进行对齐
                      否则使用人脸检测来获取人脸
            face_idx: 如果检测到多个人脸，指定使用第几个人脸 (默认使用第一个)
            
        Returns:
            embedding: 人脸特征向量
            aligned_face: 对齐后的人脸图像
        """
        if image is None:
            print("警告: 输入图像为None")
            return None, None
            
        try:
            img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            
            # 如果没有提供landmark，则检测人脸
            if landmark is None:
                faces = self.face_analysis.get(img_rgb)
                if len(faces) == 0:
                    print("未检测到人脸")
                    return None, None
                    
                # 获取指定索引的人脸
                if face_idx >= len(faces):
                    face_idx = 0  # 如果索引超出范围，使用第一个人脸
                face = faces[face_idx]
                
                # 直接使用InsightFace提供的embedding
                embedding = face.embedding
                aligned_face = face.embedding_norm
                
                # 绘制人脸边界框和关键点
                result_img = image.copy()
                bbox = face.bbox.astype(int)
                cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # 绘制关键点
                if hasattr(face, 'kps'):
                    kps = face.kps.astype(np.int32)
                    for i in range(5):
                        cv2.circle(result_img, (kps[i][0], kps[i][1]), 1, (0, 0, 255), 2)
                
                return embedding, result_img
            else:
                # 使用提供的landmark进行人脸对齐
                aligned_face = norm_crop(image, landmark)
                
                # 使用InsightFace的ArcFace模型提取特征
                if hasattr(self, 'model'):
                    # 如果已经加载了自定义模型，使用它
                    embedding = self.model.get(aligned_face)
                else:
                    # 否则使用预处理后的图像通过FaceAnalysis获取embedding
                    # 这部分需要根据您的实际情况调整
                    aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    faces = self.face_analysis.get(aligned_face_rgb)
                    if len(faces) > 0:
                        embedding = faces[0].embedding
                    else:
                        print("对齐后的人脸无法提取特征")
                        return None, aligned_face
                
                return embedding, aligned_face
                
        except Exception as e:
            print(f"提取人脸特征时发生错误: {str(e)}")
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
            embedding1, face1 = self.get_face_embedding(image1)
            if embedding1 is None:
                return -1.0, image1
                
            # 获取第二张图像的人脸特征
            embedding2, face2 = self.get_face_embedding(image2)
            if embedding2 is None:
                return -1.0, image2
                
            # 计算相似度
            similarity = self.compute_similarity(embedding1, embedding2)
            
            # 创建结果可视化
            h1, w1 = face1.shape[:2]
            h2, w2 = face2.shape[:2]
            result_img = np.zeros((max(h1, h2), w1 + w2 + 100, 3), dtype=np.uint8)
            result_img[:h1, :w1] = face1
            result_img[:h2, w1+100:w1+100+w2] = face2
            
            # 添加相似度文本
            text = f"相似度: {similarity:.4f}"
            cv2.putText(result_img, text, (w1//2, h1+30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 255, 255), 2)
                        
            return similarity, result_img
            
        except Exception as e:
            print(f"比较人脸时发生错误: {str(e)}")
            return -1.0, None


# 连接到本地Ray集群
# 初始化Ray - 如果已经初始化过就忽略这个错误
ray.init(address="auto", ignore_reinit_error=True)

# 尝试终止已有的Actor
try:
    actor = ray.get_actor("ArcFace", namespace="face_recognition")
    ray.kill(actor)
    print(f"已终止Actor: ArcFace")
except:
    print(f"Actor ArcFace 不存在或已终止")

# 部署新的Actor
deploy = ArcFaceRecognition.options(
    name="ArcFace",
    namespace="face_recognition",
    lifetime="detached",
    max_restarts=-1,
    num_cpus=1,
    num_gpus=0
).remote()

print("ArcFace已部署")