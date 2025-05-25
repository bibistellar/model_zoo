import cv2
import matplotlib.pyplot as plt
import numpy as np
import ray
import os
import time
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='OpenPose使用Ray部署进行人体姿态检测')
    parser.add_argument('--image', type=str, default='COCO_val2014_000000000192.jpg', 
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default='openpose_result.png', 
                        help='输出图像路径')
    args = parser.parse_args()
    
    # 连接到Ray集群
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except:
        print("无法连接到Ray集群，将创建本地Ray实例...")
        ray.init()
    
    # 连接到OpenPose Actor
    try:
        pose_detector = ray.get_actor("OpenPose", namespace="pose_detection")
        print("成功连接到OpenPose姿态检测Actor")
    except:
        print("无法找到OpenPose Actor，请确保先运行ray_openpose.py")
        exit(1)
    
    # 检查输入图像是否存在
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print(f"当前工作目录: {os.getcwd()}")
        exit(1)
    
    # 输出图像路径
    output_path = args.output
    
    print(f"正在处理图像: {image_path}")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像文件: {image_path}")
        exit(1)
    
    # 开始计时
    start_time = time.time()
    
    # 调用OpenPose进行姿态检测
    result = ray.get(pose_detector.detect_pose.remote(img))
    processed_img, keypoints = result
    
    # 结束计时
    end_time = time.time()
    print(f"总处理时间: {end_time - start_time:.4f} 秒")
    
    if processed_img is None:
        print("处理失败，未返回图像结果")
        exit(1)
    
    # 保存结果图像
    cv2.imwrite(output_path, processed_img)
    print(f"结果已保存到: {output_path}")
    
    # 打印检测到的人体数量
    print(f"检测到 {len(keypoints)} 个人体")
    
    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('OpenPose Human Pose Detection')
    plt.show()

if __name__ == "__main__":
    main()
