# filepath: /root/Code/aiplat/model_zoo/main.py
import ray
import os
import logging
# from ray_buffalo_l import BuffaloFaceDetector

def init_ray_local_cluster(num_cpus=None, num_gpus=None, dashboard_port=8265):
    """
    初始化本地Ray集群
    
    Args:
        num_cpus: 分配的CPU数量，默认为None（使用所有可用的CPU）
        num_gpus: 分配的GPU数量，默认为None（使用所有可用的GPU）
        dashboard_port: Ray仪表板端口，默认为8265
    """
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Ray集群")
    
    try:
        # 初始化Ray集群
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            dashboard_port=dashboard_port,
            include_dashboard=True,
            ignore_reinit_error=True,
        )
        logger.info("本地Ray集群已启动")
        logger.info(f"Ray Dashboard可以通过 http://localhost:{dashboard_port} 访问")
        
        # 创建并注册BuffaloFaceDetector Actor
        # face_detector = BuffaloFaceDetector.remote()
        # ray.get(face_detector.print_sysytem_info.remote())  # 打印系统信息
        # ray.util.register_actor("Buffalo", face_detector, namespace="face_detection")
        # logger.info("人脸检测服务已注册，可以通过 'Buffalo' 名称访问")
        
        return
    except Exception as e:
        logger.error(f"启动Ray集群失败: {e}")
        raise

def main():
    """
    主函数：启动本地Ray集群，并保持运行
    """
    import time
    
    print("正在启动本地Ray集群...")
    init_ray_local_cluster()
    print("集群已启动并运行中。按Ctrl+C退出。")
    
    try:
        # 保持程序运行，直到手动中断
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("正在关闭Ray集群...")
        ray.shutdown()
        print("Ray集群已关闭")

if __name__ == "__main__":
    main()