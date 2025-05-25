# RetinaFace Ray Actor

这个模块将RetinaFace人脸检测模型封装为Ray Actor，使其可以在Ray集群上进行分布式部署和调用。

## 功能特点

- 基于Ray的分布式部署
- 支持TPU加速
- 支持人脸检测和关键点定位（如果模型支持）
- 提供可视化工具
- 易于集成到现有系统

## 依赖环境

- Python 3.8+
- Ray 2.0+
- OpenCV 4.0+
- Sophon SAIL (用于TPU加速)
- NumPy

## 快速开始

1. 确保已安装所有依赖项：

```bash
pip install ray opencv-python numpy
# 安装sophon sail请参考官方文档
```

2. 启动Ray：

```bash
ray start --head --port=10001
```

3. 运行示例：

```bash
python run.py --model /path/to/your/retinaface.bmodel --image /path/to/your/image.jpg
```

## 脚本参数说明

```
--model       模型文件路径(.bmodel)
--image       输入图像路径
--output      输出图像路径 (默认: result.jpg)
--tpu-id      TPU设备ID (默认: 0)
--threshold   检测阈值 (默认: 0.5)
--nms         NMS阈值 (默认: 0.4)
--max-faces   最大检测人脸数，0表示不限制 (默认: 0)
--ray-address Ray集群地址 (默认: auto)
```

## 代码中使用

```python
import ray
import cv2
from ray_retinaFace import RetinaFaceDetector

# 初始化Ray
ray.init(address="auto")

# 创建RetinaFace Actor
detector = RetinaFaceDetector.remote(
    model_file="/path/to/your/retinaface.bmodel", 
    tpu_id=0,
    nms_thresh=0.4,
    det_thresh=0.5
)

# 读取图像
image = cv2.imread("test.jpg")

# 检测人脸
det, kpss = ray.get(detector.detect_face.remote(image))

# 可视化结果
result_image = ray.get(detector.visualize.remote(image, det, kpss))

# 保存结果
cv2.imwrite("result.jpg", result_image)
```

## RetinaFaceDetector API

### 初始化

```python
detector = RetinaFaceDetector.remote(
    model_file=None,  # 模型文件路径
    tpu_id=0,         # TPU设备ID
    nms_thresh=0.4,   # 非极大值抑制阈值
    det_thresh=0.5    # 检测阈值
)
```

### 加载模型

```python
success = ray.get(detector.load_model.remote(
    model_file="path/to/model.bmodel",
    tpu_id=0,
    nms_thresh=0.4,
    det_thresh=0.5
))
```

### 检测人脸

```python
det, kpss = ray.get(detector.detect_face.remote(
    image,           # 输入图像
    max_num=0,       # 最大检测人脸数，0表示不限制
    metric='default' # 排序方式，'default'或'max'
))
```

### 可视化结果

```python
result_image = ray.get(detector.visualize.remote(
    image,  # 原始图像
    det,    # 检测结果
    kpss    # 关键点 (可选)
))
```

### 运行演示

```python
result_image, det, kpss = ray.get(detector.run_demo.remote("test.jpg"))
```

## 注意事项

1. 确保模型文件(.bmodel)存在且有效
2. TPU设备需要正确配置和安装驱动
3. 在生产环境中，可能需要调整Ray Actor的配置，如CPU、内存、资源限制等
4. 建议使用`lifetime="detached"`参数，使Actor在脚本结束后仍然存活，提高复用效率
