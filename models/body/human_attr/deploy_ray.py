import ray
from ray_body_detect import BodyModel
from runtime_config import body_conda_env, ray_address, bodydetect_config

# 连接到本地Ray集群
# 注意：请确保先运行了main.py启动本地集群
# 连接到本地Ray集群
# 初始化Ray - 如果已经初始化过就忽略这个错误
ray.init(address=ray_address, ignore_reinit_error=True)

# 尝试终止已有的Actor
try:
    actor = ray.get_actor(bodydetect_config["name"], namespace=bodydetect_config["namespace"])
    ray.kill(actor)
    print(f"已终止Actor: BodyModel")
except:
    print(f"Actor BodyModel 不存在或已终止")

# 部署新的Actor - 使用正确的runtime_env配置
deploy = BodyModel.options(
    name=bodydetect_config["name"],
    namespace=bodydetect_config["namespace"],
    lifetime="detached",
    max_restarts=-1,
    num_cpus=1,
    num_gpus=0,
    runtime_env=body_conda_env  # 确保使用paddle环境
).remote()

print("BodyModel已部署到py3.8_paddle环境")

# 测试部署是否成功
try:
    health_status = ray.get(deploy.health_check.remote())
    print("健康检查结果:")
    print(health_status)
except Exception as e:
    print(f"健康检查失败: {e}")
    print("请确保py3.8_paddle conda环境已正确配置并包含paddle依赖")

print("BodyModel已部署")