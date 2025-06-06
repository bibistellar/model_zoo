import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# pipeline_dir = os.path.join(current_dir, 'pipeline')
# if pipeline_dir not in sys.path:
#     sys.path.insert(0, pipeline_dir)
import ray

import numpy as np
from runtime_config import body_conda_env

@ray.remote(num_cpus=1, num_gpus=0, runtime_env=body_conda_env)
class BodyModel:
    def __init__(self, device='CPU'):
        # 延迟导入：在Actor实际运行的conda环境中导入paddle相关依赖
        self.device = device
        self.pipeline = None
        
    def _ensure_initialized(self):
        """确保模型已初始化，使用延迟加载模式"""
        if self.pipeline is None:
            try:
                # 在运行时环境中导入依赖
                import paddle
                from pipeline.cfg_utils import argsparser, print_arguments, merge_cfg
                from pipeline.pipeline import Pipeline
                
                parser = argsparser()
                # 使用空参数列表避免解析Ray的内部参数
                FLAGS = parser.parse_args([])
                FLAGS.config = os.path.join(current_dir,"pipeline/config/examples/infer_cfg_human_attr.yml")
                paddle.enable_static()
                FLAGS.device = self.device
                assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU', 'GCU'], \
                    "device should be CPU, GPU, XPU, NPU or GCU"
                cfg = merge_cfg(FLAGS)  # use command params to update config
                print_arguments(cfg)
                
                self.pipeline = Pipeline(FLAGS, cfg)
                print(f"BodyModel 初始化成功，使用设备: {self.device}")
                
            except ImportError as e:
                raise RuntimeError(f"无法导入paddle相关依赖: {e}. 请确保在正确的conda环境中运行")
            except Exception as e:
                raise RuntimeError(f"初始化BodyModel失败: {e}")

    def getresult(self, img):
        self._ensure_initialized()
        res = self.pipeline.run(img)
        print("res:", res)
        rd = res.get('res_dict', {})

        det = rd.get('det', {})
        boxes_arr = det.get('boxes', np.zeros((0, 6), dtype=np.float32))
        boxes = []
        for b in boxes_arr:
            x_min, y_min, x_max, y_max = b[2], b[3], b[4], b[5]
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            boxes.append([x, y, w, h])

        attr = rd.get('attr', {})
        attrs = attr.get('output', [])

        return boxes, attrs
    
    def getcarresult(self, img):
        self._ensure_initialized()
        res = self.pipeline.run(img)
        rd = res.get('res_dict', {})

        det = rd.get('det', {})
        boxes_arr = det.get('boxes', np.zeros((0, 6), dtype=np.float32))
        boxes = []
        for b in boxes_arr:
            x_min, y_min, x_max, y_max = b[2], b[3], b[4], b[5]
            x, y = int(x_min), int(y_min)
            w, h = int(x_max - x_min), int(y_max - y_min)
            boxes.append([x, y, w, h])

        vehicleplate = rd.get('vehicleplate', {}).get('vehicleplate', [])
        vehicle_attr = rd.get('vehicle_attr', {}).get('output', [])

        return boxes, vehicleplate, vehicle_attr
    
    def health_check(self):
        """健康检查：验证paddle环境是否可用"""
        try:
            # 检查paddle导入
            import paddle
            paddle_version = paddle.__version__
            
            # 检查pipeline模块
            from pipeline.cfg_utils import argsparser
            from pipeline.pipeline import Pipeline
            
            return {
                "status": "healthy",
                "paddle_version": paddle_version,
                "device": self.device,
                "initialized": self.pipeline is not None
            }
        except ImportError as e:
            return {
                "status": "error",
                "error": f"依赖导入失败: {str(e)}",
                "suggestion": "请确保在py3.8_paddle conda环境中运行"
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": f"其他错误: {str(e)}"
            }