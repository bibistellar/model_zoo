import cv2
from matplotlib import pyplot as plt
import numpy as np
import ray
import argparse
import os
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Buffalo-L人脸检测样例")
    parser.add_argument('--image', type=str, default="./image1.png", help="输入图像路径")
    parser.add_argument('--output', type=str, default="./detected_face.png", help="输出图像路径")
    parser.add_argument('--show', action="store_true", help="显示检测结果")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 连接到本地Ray集群
    ray.init(ignore_reinit_error=True)  # 连接到本地集群
    
    # 获取人脸检测Actor
    face_detector = ray.get_actor("Buffalo", namespace="face_detection")
    
    # 检查输入图像是否存在
    if not os.path.exists(args.image):
        print(f"错误: 输入图像 '{args.image}' 不存在")
        sys.exit(1)
        
    print(f"正在处理图像: {args.image}")
    
    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"错误: 无法读取图像 '{args.image}'")
        sys.exit(1)
    
    # 调用远程服务进行人脸检测
    new_img = ray.get(face_detector.detect_face.remote(img))
    
    # 保存结果
    cv2.imwrite(args.output, new_img)
    print(f"检测结果已保存到: {args.output}")
    
    # # 如果需要，显示结果
    # if args.show:
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    #     plt.title("检测到的人脸")
    #     plt.axis('off')
    #     plt.show()