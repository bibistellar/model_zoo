import platform
import sys
import cv2
import numpy as np
import ray
from ray.runtime_env import RuntimeEnv
import os
import onnxruntime
from PIL import Image

# 统一的conda环境配置
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class ReIDFeatureExtractor:
    """Ray Actor 封装人体ReID特征提取功能"""
    
    def __init__(self, model_path=None):
        """
        初始化ReID特征提取器 - 仅支持ONNX模型，CPU推理
        
        Args:
            model_path: ONNX模型文件路径，如果为None则使用默认路径
        """
        # 默认加载当前目录下的resnet50模型
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "resnet50_reid.onnx")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX模型文件不存在: {model_path}")
        
        # 加载ONNX模型 - 仅使用CPU
        self.model = self._load_onnx_model(model_path)
        
        print(f"ReID特征提取器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"特征维度: {self.feature_dim}")

    def _load_onnx_model(self, model_file):
        """加载ONNX模型 - 仅使用CPU"""
        try:
            # 仅使用CPU执行
            session = onnxruntime.InferenceSession(model_file, providers=['CPUExecutionProvider'])
            
            # 获取输入输出信息
            self.input_name = session.get_inputs()[0].name
            self.input_shape = session.get_inputs()[0].shape
            self.output_name = session.get_outputs()[0].name
            self.output_shape = session.get_outputs()[0].shape
            
            # 推断特征维度
            if len(self.output_shape) >= 2:
                self.feature_dim = self.output_shape[-1]
            else:
                self.feature_dim = 2048  # 默认值
            
            print(f"ONNX模型输入形状: {self.input_shape}")
            print(f"ONNX模型输出形状: {self.output_shape}")
            
            return session
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {str(e)}")

    def get_model_info(self):
        """获取模型信息"""
        return {
            "feature_dim": self.feature_dim,
            "input_size": (224, 224),
            "device": "CPU"
        }

    def print_system_info(self):
        """打印系统信息"""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "onnxruntime_version": onnxruntime.__version__
        }

    def extract_feature(self, crop):
        """
        提取单张人体图像的ReID特征向量
        
        Args:
            crop: 人体裁剪图像 (BGR格式的numpy数组)
            
        Returns:
            feature: L2归一化后的特征向量 (numpy.ndarray, shape: [feature_dim])
        """
        if crop is None or crop.size == 0:
            raise ValueError("输入图像无效")
        
        try:
            # BGR转RGB
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # 图像预处理
            from PIL import Image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((224, 224))  # 标准ImageNet输入尺寸 (width, height)
            
            # 转换为numpy数组并标准化
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            img_array = (img_array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # 调整维度 (H, W, C) -> (1, C, H, W)
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 确保数据类型为float32
            img_array = img_array.astype(np.float32)
            
            # ONNX推理
            feat = self.model.run([self.output_name], {self.input_name: img_array})[0]
            feat = feat.squeeze()
            
            # L2归一化
            feat = feat / np.linalg.norm(feat)
            
            return feat
            
        except Exception as e:
            raise RuntimeError(f"特征提取失败: {str(e)}")

# 如果直接运行此脚本，进行简单测试
if __name__ == "__main__":
    # 初始化Ray
    ray.init(ignore_reinit_error=True)
    
    # 创建ReID特征提取器
    extractor = ReIDFeatureExtractor.remote()
    
    # 打印系统信息
    system_info = ray.get(extractor.print_system_info.remote())
    print("系统信息:", system_info)
    
    # 获取模型信息
    model_info = ray.get(extractor.get_model_info.remote())
    print("模型信息:", model_info)
    
    print("ReID特征提取器Ray Actor测试完成")
