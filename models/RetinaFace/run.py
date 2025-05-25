from Cython import address
import cv2
from matplotlib import pyplot as plt
import numpy as np
import ray
import argparse
import os
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RetinaFace人脸检测样例")
    parser.add_argument('--image', type=str, default="./image1.png", help="输入图像路径")
    parser.add_argument('--output', type=str, default="./detected_face.png", help="输出图像路径")
    parser.add_argument('--align', action="store_true", help="是否提取对齐的人脸")
    parser.add_argument('--align_output_dir', type=str, default="./aligned_faces", help="对齐人脸的输出目录")
    parser.add_argument('--show', action="store_true", help="显示检测结果")
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 连接到本地Ray集群
    ray.init(address="auto",ignore_reinit_error=True)  # 连接到本地集群
    
    # 获取人脸检测Actor
    try:
        face_detector = ray.get_actor("RetinaFace", namespace="face_detection")
    except:
        print("错误: 未找到RetinaFace服务，请确保先运行了部署脚本")
        sys.exit(1)
    
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
    result_img, faces = ray.get(face_detector.detect_face.remote(image=img))
    
    if result_img is not None:
        # 保存结果
        cv2.imwrite(args.output, result_img)
        print(f"检测结果已保存到: {args.output}")
        
        # 如果需要，显示结果
        if args.show:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title(f"检测到 {len(faces)} 个人脸")
            plt.axis('off')
            plt.show()
        
        # 检测到的人脸信息
        print(f"检测到 {len(faces)} 个人脸")
        for face_id, face_info in faces.items():
            confidence = face_info['score']
            bbox = face_info['facial_area']
            print(f"- {face_id}: 置信度={confidence:.4f}, 边界框={bbox}")
    else:
        print("未能成功处理图像")
    
    # 如果需要提取对齐的人脸
    if args.align:
        # 确保输出目录存在
        if not os.path.exists(args.align_output_dir):
            os.makedirs(args.align_output_dir)
        
        aligned_faces = ray.get(face_detector.extract_aligned_faces.remote(image=img))
        
        if aligned_faces:
            print(f"提取了 {len(aligned_faces)} 个对齐的人脸")
            
            # 保存对齐的人脸
            for i, face in enumerate(aligned_faces):
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if len(face.shape) == 3 and face.shape[2] == 3 else face
                face_filename = os.path.join(args.align_output_dir, f"face_{i+1}.jpg")
                plt.imsave(face_filename, face_rgb)
                print(f"对齐的人脸已保存到: {face_filename}")
            
            # 如果需要，显示对齐的人脸
            if args.show and aligned_faces:
                fig, axes = plt.subplots(1, len(aligned_faces), figsize=(3*len(aligned_faces), 3))
                if len(aligned_faces) == 1:
                    axes = [axes]
                for i, face in enumerate(aligned_faces):
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if len(face.shape) == 3 and face.shape[2] == 3 else face
                    axes[i].imshow(face_rgb)
                    axes[i].axis('off')
                    axes[i].set_title(f"Face {i+1}")
                plt.tight_layout()
                plt.show()
        else:
            print("未能提取对齐的人脸")