import cv2
from matplotlib import pyplot as plt
import numpy as np
import ray
import argparse
import os
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ArcFace人脸识别样例")
    parser.add_argument('--image1', type=str, required=True, help="第一张人脸图像路径")
    parser.add_argument('--image2', type=str, required=True, help="第二张人脸图像路径")
    parser.add_argument('--output', type=str, default="./arcface_comparison.png", help="输出图像路径")
    parser.add_argument('--threshold', type=float, default=0.5, help="人脸识别相似度阈值，超过此阈值认为是同一个人")
    parser.add_argument('--model', type=str, default=None, help="自定义ArcFace模型路径，为空则使用InsightFace默认模型")
    parser.add_argument('--show', action="store_true", help="显示检测结果")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 连接到本地Ray集群
    ray.init(address="auto", ignore_reinit_error=True)
    
    # 获取人脸识别Actor
    try:
        face_recognizer = ray.get_actor("ArcFace", namespace="face_recognition")
        print("成功连接到ArcFace Actor")
    except:
        print("错误：未找到ArcFace Actor，请确保先运行ray_arcface.py")
        sys.exit(1)
    
    # 如果提供了自定义模型，则加载它
    if args.model is not None and os.path.exists(args.model):
        success = ray.get(face_recognizer.load_model.remote(model_file=args.model))
        if not success:
            print(f"警告: 无法加载模型 '{args.model}'，将使用默认模型")
    
    # 检查输入图像是否存在
    for img_path in [args.image1, args.image2]:
        if not os.path.exists(img_path):
            print(f"错误: 输入图像 '{img_path}' 不存在")
            sys.exit(1)
    
    print(f"正在比较人脸图像: \n  {args.image1}\n  {args.image2}")
    
    # 读取图像
    img1 = cv2.imread(args.image1)
    img2 = cv2.imread(args.image2)
    
    if img1 is None or img2 is None:
        print("错误: 无法读取输入图像")
        sys.exit(1)
    
    # 调用远程服务比较人脸
    similarity, result_img = ray.get(face_recognizer.compare_faces.remote(img1, img2))
    
    if similarity < 0:
        print("错误: 未能成功比较人脸，可能原因是未检测到人脸")
    else:
        print(f"人脸相似度: {similarity:.4f}")
        
        if similarity > args.threshold:
            print("结果: 同一个人")
        else:
            print("结果: 不同的人")
    
        # 保存结果
        if result_img is not None:
            cv2.imwrite(args.output, result_img)
            print(f"比较结果已保存到: {args.output}")
            
            # 如果需要，显示结果
            if args.show and result_img is not None:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                plt.title(f"人脸相似度: {similarity:.4f}")
                plt.axis('off')
                plt.show()
