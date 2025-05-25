import ray
from ray_arcface_onnx import ArcFaceRecognition
from runtime_config import conda_env,ray_address, arcface_config
# 连接到本地Ray集群
# 注意：请确保先运行了main.py启动本地集群
# 连接到本地Ray集群
# 初始化Ray - 如果已经初始化过就忽略这个错误
ray.init(address=ray_address, ignore_reinit_error=True)

# 尝试终止已有的Actor
try:
    actor = ray.get_actor(arcface_config["name"], namespace=arcface_config["namespace"])
    ray.kill(actor)
    print(f"已终止Actor: ArcFace")
except:
    print(f"Actor ArcFace 不存在或已终止")

# 部署新的Actor
deploy = ArcFaceRecognition.options(
    name=arcface_config["name"],
    namespace=arcface_config["namespace"],
    lifetime="detached",
    max_restarts=-1,
    num_cpus=1,
    num_gpus=0
).remote()

print("ArcFace已部署")