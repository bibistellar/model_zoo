# PaddleDetection 人体属性检测模型

## 目录说明

本目录下的文件夹主要来自 **PaddleDetection** 的 runtime 代码，用于人体属性检测模型的部署和推理。该模型能够检测图像中的人体并识别其属性特征，包括性别、年龄、姿态、服饰等多维度信息。这些代码在原有基础上进行了小的修改，以适配当前项目的需求。

## 模型功能

人体属性检测模型能够识别以下属性：
- **性别识别**：男/女
- **年龄估计**：18-60岁等年龄段
- **姿态分析**：正面/背面/侧面
- **配饰检测**：眼镜、帽子等
- **携带物品**：包、手持物品等
- **服装分析**：上衣类型（长袖/短袖）、下装类型（长裤/短裤/裙子等）
- **鞋类识别**：靴子/非靴子

## 输出格式

模型输出包含两部分：
1. **检测框坐标**：人体在图像中的位置 `[x1, y1, x2, y2]`
2. **属性列表**：包含各类属性的识别结果

### 输出示例
```python
# 检测框坐标（左上角和右下角坐标）
[[0, 33, 278, 232]]

# 属性识别结果
[['男', '18-60岁', '正面', '戴眼镜: 是', '戴帽子: 否', '正面持物: 否', '无包', '上衣: 长袖', '下衣: 长款大衣 长裤', '非靴子']]
```

## 目录结构

- `auto_compression/` - 模型自动压缩相关代码（来自PaddleDetection）
- `benchmark/` - 性能基准测试工具（来自PaddleDetection）  
- `cpp/` - C++部署实现（来自PaddleDetection）
- `end2end_ppyoloe/` - PP-YOLOE端到端部署（来自PaddleDetection）
- `fastdeploy/` - FastDeploy部署方案（来自PaddleDetection）
- `lite/` - Paddle Lite移动端部署（来自PaddleDetection）
- `pipeline/` - 检测流水线实现（来自PaddleDetection）
- `pptracking/` - PP-Tracking多目标跟踪（来自PaddleDetection）
- `python/` - Python部署实现（来自PaddleDetection）
- `serving/` - 模型服务化部署（来自PaddleDetection）
- `third_engine/` - 第三方推理引擎支持（来自PaddleDetection）

## 项目定制文件

- `deploy_ray.py` - Ray分布式部署脚本（项目定制）
- `ray_body_detect.py` - Ray身体检测实现（项目定制）
- `test_body_actor.py` - 身体检测Actor测试（项目定制）

## 使用说明

这些runtime代码提供了多种部署方式，包括：
- **Python推理部署** - 适用于开发和测试环境
- **C++高性能部署** - 适用于生产环境，性能更优
- **移动端轻量化部署** - 适用于移动设备和边缘计算
- **服务化在线部署** - 适用于Web服务和API接口
- **分布式Ray部署** - 适用于大规模并发处理

### 快速开始

1. 使用Ray分布式部署：
```bash
python deploy_ray.py
```

详细使用方法请参考各子目录下的README文档。

## 来源

大部分代码来源于 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 项目的 deploy 模块，在此基础上针对项目需求进行了适配和优化。