"""
Ray 运行时环境配置文件
"""
from ray.runtime_env import RuntimeEnv

#ray_address = "ray://localhost:10001"  # Ray集群地址
ray_address = "auto"  # Ray集群地址
# 统一的conda环境配置
conda_env = RuntimeEnv(
    conda="py3.8_ray"
)

body_conda_env = RuntimeEnv(
    conda="py3.8_paddle"
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

# genderage相关配置
genderage_config = {
    "name": "Genderage",
    "namespace": "face_detection",
    "lifetime": "detached",
    "max_restarts": -1,
    "num_cpus": 1,
    "num_gpus": 0,
    "runtime_env": conda_env
}

# bodydetect相关配置
bodydetect_config = {
    "name": "bodydetect",
    "namespace": "body_detection",
    "lifetime": "detached",
    "max_restarts": -1,
    "num_cpus": 1,
    "num_gpus": 0,
    "runtime_env": body_conda_env
}

# ReID特征提取器相关配置
reid_config = {
    "name": "ReIDExtractor",
    "namespace": "body_detection",
    "lifetime": "detached",
    "max_restarts": -1,
    "num_cpus": 2,
    "num_gpus": 0,  # 如果有GPU则使用GPU
    "runtime_env": conda_env
}

# # 人体检测和裁剪相关配置
# person_detector_config = {
#     "name": "PersonDetector",
#     "namespace": "person_detection",
#     "lifetime": "detached",
#     "max_restarts": -1,
#     "num_cpus": 1,
#     "num_gpus": 0,
#     "runtime_env": person_detect_conda_env
# }