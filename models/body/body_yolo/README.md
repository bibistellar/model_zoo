# Body YOLO 人体检测模块

本目录存放使用YOLO模型进行人体检测的代码，提供了两种不同的推理方式以及结果对比工具。

## 目录说明

该模块实现了基于YOLO的人体检测功能，主要用于定位图像中的人体并提供边界框坐标。由于功能与 `human_attr` 模块中的人体定位功能重复，因此暂时只提供测试代码，未包含部署相关代码。

## 文件结构

```
body_yolo/
├── README.md                    # 本说明文档
├── detect_and_crop.py          # 基于ultralytics的人体检测实现
├── detect_and_crop_onnx.py     # 基于ONNX推理的人体检测实现
└── compare_results.py          # 两种实现方式的结果对比工具
```

## 功能特性

### 1. Ultralytics实现 (`detect_and_crop.py`)
- **框架**: 使用ultralytics库
- **模型**: YOLO11n.pt
- **优势**: 
  - 使用简便，API友好
  - 自动处理模型加载和预处理
  - 内置优化和后处理逻辑

### 2. ONNX推理实现 (`detect_and_crop_onnx.py`)
- **框架**: 使用onnxruntime
- **模型**: YOLO11n.onnx
- **优势**:
  - 更好的跨平台兼容性
  - 可控的推理过程
  - 更好的部署灵活性
- **特性**:
  - 自定义预处理(尺寸调整、填充、归一化)
  - 手动实现的后处理和NMS
  - 支持置信度和IoU阈值配置

### 3. 结果对比工具 (`compare_results.py`)
- 同时运行两种实现方式
- 比较检测结果的一致性
- 计算边界框的IoU相似度
- 提供详细的对比分析报告

## 使用方法

### 基本使用

```python
# 使用ultralytics版本
from detect_and_crop import PersonDetector

detector = PersonDetector()
results = detector.detect_and_crop(frame)

# 使用ONNX版本  
from detect_and_crop_onnx import PersonDetectorONNX

detector_onnx = PersonDetectorONNX()
results_onnx = detector_onnx.detect_and_crop(frame)
```

### 运行对比测试

```bash
python compare_results.py
```

### 单独测试

```bash
# 测试ultralytics版本
python detect_and_crop.py

# 测试ONNX版本
python detect_and_crop_onnx.py
```

## 输出格式

两种实现都返回相同格式的检测结果:

```python
[
    {
        'crop': numpy.ndarray,           # 裁剪出的人体图像
        'bbox': [x1, y1, x2, y2],      # 边界框坐标
        'score': float                   # 置信度分数(仅ONNX版本)
    },
    ...
]
```

## 配置参数

### PersonDetector (ultralytics)
- `model_path`: 模型文件路径 (默认: 'yolov11n.pt')
- `conf_thresh`: 置信度阈值 (默认: 0.5)

### PersonDetectorONNX (ONNX)
- `model_path`: ONNX模型文件路径 (默认: '/root/Code/aiplat/model_zoo/reid/yolo11n.onnx')
- `conf_thresh`: 置信度阈值 (默认: 0.5)
- `iou_thresh`: IoU阈值用于NMS (默认: 0.45)

## 依赖要求

```
opencv-python
numpy
ultralytics
onnxruntime
```

## 注意事项

1. **功能重复**: 该模块与 `human_attr` 中的人体定位功能存在重复，主要用于测试和比较不同实现方式
2. **模型路径**: 确保ONNX模型文件路径正确，默认指向 `/root/Code/aiplat/model_zoo/reid/yolo11n.onnx`
3. **图像格式**: 输入图像需要是BGR格式的numpy数组
4. **坐标系统**: 返回的边界框坐标为 (x1, y1, x2, y2) 格式，其中 (x1, y1) 为左上角，(x2, y2) 为右下角

## 性能对比

| 实现方式 | 优势 | 劣势 |
|---------|------|------|
| Ultralytics | 简单易用，内置优化 | 依赖较重，定制性较低 |
| ONNX | 轻量化，跨平台，可定制 | 需要手动实现预后处理 |

## 后续计划

由于与 `human_attr` 模块功能重复，该模块暂时作为测试和比较工具使用。如需正式部署人体检测功能，建议：

1. 整合到 `human_attr` 模块中
2. 或者作为独立的人体检测服务进行优化部署
3. 根据实际需求选择合适的推理方式
