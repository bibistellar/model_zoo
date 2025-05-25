"""
Ray 运行时环境配置文件
"""

import ray
from ray.runtime_env import RuntimeEnv

from models import buffalo_l

#ray_address = "ray://localhost:10001"  # Ray集群地址
ray_address = "auto"  # Ray集群地址
# 统一的conda环境配置
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

# buffalo_l相关配置
buffalo_l_config = {
    "name": "Buffalo",
    "namespace": "face_detection",
    "lifetime": "detached",
    "max_restarts": -1,
    "num_cpus": 1,
    "num_gpus": 0,
    "runtime_env": conda_env
}

# arcface相关配置
arcface_config = {
    "name": "Arcface",
    "namespace": "face_detection",
    "lifetime": "detached",
    "max_restarts": -1,
    "num_cpus": 1,
    "num_gpus": 0,
    "runtime_env": conda_env
}