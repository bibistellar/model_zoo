import ray
import sys
import os

# 添加当前目录和项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)  # 添加当前目录到路径最前面
sys.path.append(project_root)

from ray_buffalo_l_onnx import BuffaloFaceDetector
from runtime_config import conda_env, ray_address, buffalo_l_config


def main():
    """部署Buffalo人脸检测器"""
    # 连接到本地Ray集群
    # 注意：请确保先运行了main.py启动本地集群
    ray.init(address=ray_address, ignore_reinit_error=True)  # 连接到本地集群，无需指定地址
    
    try:
        actor = ray.get_actor(buffalo_l_config["name"], namespace=buffalo_l_config["namespace"])
        ray.kill(actor)
        print(f"已终止Actor: Buffalo")
    except:
        print(f"Actor Buffalo 不存在或已终止")

    # 部署新的Actor
    deploy = BuffaloFaceDetector.options(
        name=buffalo_l_config["name"],
        namespace=buffalo_l_config["namespace"],
        lifetime="detached",
        max_restarts=-1,
        num_cpus=1,
        num_gpus=0
    ).remote()
    print("Buffalo已部署")


if __name__ == "__main__":
    main()