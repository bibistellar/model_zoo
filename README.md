# Face Detection Service with Ray and InsightFace Buffalo-L

这个项目使用Ray分布式计算框架和InsightFace的Buffalo-L模型实现了高性能的人脸检测服务。

## 环境要求

### Conda环境配置

1. 创建conda环境:

```bash
conda create -n face_detection python=3.8
conda activate face_detection
```

2. 安装必要的依赖:

```bash
# 安装基本依赖
pip install numpy opencv-python matplotlib

# 安装Ray分布式计算框架
pip install ray[default]

# 安装InsightFace (Buffalo-L模型的依赖)
pip install insightface onnxruntime-gpu

# 如果需要GPU加速，请确保已安装CUDA和cuDNN
```

## 项目结构

```
model_zoo/
├── main.py                 # Ray集群启动文件
├── README.txt              # 项目说明文档
└── models/
    └── buffalo_l/
        ├── buffalo_l.py    # Buffalo-L模型基础实现
        ├── ray_buffalo_l.py # Ray Actor封装的模型实现
        └── run.py          # 调用示例
```

## 使用说明

### 1. 启动Ray集群

首先，启动本地Ray集群：

```bash
cd /root/Code/aiplat/model_zoo
python main.py
```

这将启动一个本地Ray集群，并在控制台输出Ray Dashboard的访问地址（默认为http://localhost:8265）。

### 2. 部署人脸检测服务

在另一个终端窗口中，部署Buffalo-L人脸检测服务到Ray集群：

```bash
cd /root/Code/aiplat/model_zoo
python models/buffalo_l/ray_buffalo_l.py
```

该脚本会创建一个名为"Buffalo"的Ray Actor，并部署到集群中。

### 3. 本地验证识别效果

使用run.py进行本地验证：

```bash
cd /root/Code/aiplat/model_zoo/models/buffalo_l
python run.py --image path/to/your/image.jpg
```

这将加载指定的图像，调用Ray集群中的人脸检测服务进行处理，并显示检测结果。

### 4. 在代码中调用人脸检测服务

可以在Python代码中调用已部署的人脸检测服务：

```python
import ray
import cv2

# 连接到Ray集群
ray.init(ignore_reinit_error=True)

# 获取已部署的人脸检测Actor
face_detector = ray.get_actor("Buffalo", namespace="face_detection")

# 读取图像
image = cv2.imread("path/to/image.jpg")

# 远程调用人脸检测
result = ray.get(face_detector.detect_face.remote(image))

# 显示结果
cv2.imshow("Face Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 故障排除

1. 如果连接到Ray集群时出现问题，请确保：
   - Ray集群已经启动（运行main.py）
   - 连接地址正确（默认为本地连接）

2. 如果模型加载失败，请检查：
   - CUDA和cuDNN是否正确安装
   - InsightFace模型是否已下载

3. 如果出现资源分配错误，请检查：
   - 是否有足够的CPU/GPU资源
   - Ray集群配置是否正确

## 高级配置

### 自定义GPU资源分配

在ray_buffalo_l.py中，可以修改@ray.remote装饰器的参数来调整资源分配：

```python
@ray.remote(num_cpus=1, num_gpus=0.5)  # 分配0.5个GPU
```

### 调整检测参数

可以在BuffaloFaceDetector类的__init__方法中修改det_size参数来调整检测精度和性能：

```python
self.app.prepare(ctx_id=0, det_size=(320, 320))  # 更小的尺寸，更快的速度
```

### 分布式部署

对于多节点部署，请参考Ray官方文档配置集群地址和资源。

### 编译文件
pip install -e .