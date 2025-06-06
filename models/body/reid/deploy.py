import ray
from ray_reid_extractor import ReIDFeatureExtractor 
from runtime_config import conda_env,ray_address, reid_config
# 连接到本地Ray集群
# 注意：请确保先运行了main.py启动本地集群
# 连接到本地Ray集群
# 初始化Ray - 如果已经初始化过就忽略这个错误
ray.init(address=ray_address, ignore_reinit_error=True)

# 尝试终止已有的Actor
try:
    actor = ray.get_actor(reid_config["name"], namespace=reid_config["namespace"])
    ray.kill(actor)
    print(f"已终止Actor: ArcFace")
except:
    print(f"Actor ArcFace 不存在或已终止")

# 部署新的Actor
deploy = ReIDFeatureExtractor.options(
    name=reid_config["name"],
    namespace=reid_config["namespace"],
    lifetime="detached",
    max_restarts=-1,
    num_cpus=1,
    num_gpus=0
).remote()

print("body_embeddings模型已部署")