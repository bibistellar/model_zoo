#!/usr/bin/env python3
"""
比较原版YOLO和ONNX版本的检测结果
"""

import cv2
import numpy as np
import os
from detect_and_crop import PersonDetector
from detect_and_crop_onnx import PersonDetectorONNX

def compare_detectors():
    # 加载图像
    image_path = "/root/Code/aiplat/model_zoo/image1.png"
    if not os.path.exists(image_path):
        print(f"图像 {image_path} 不存在")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("无法加载图像")
        return
    
    print("图像尺寸:", frame.shape)
    
    # 原版YOLO检测器
    print("\n=== 原版YOLO检测器 ===")
    detector_orig = PersonDetector()
    results_orig = detector_orig.detect_and_crop(frame)
    print(f"检测到 {len(results_orig)} 个行人")
    
    for i, person in enumerate(results_orig):
        bbox = person['bbox']
        print(f"  行人 {i+1}: bbox={bbox}")
    
    # ONNX检测器
    print("\n=== ONNX检测器 ===")
    detector_onnx = PersonDetectorONNX()
    results_onnx = detector_onnx.detect_and_crop(frame)
    print(f"检测到 {len(results_onnx)} 个行人")
    
    for i, person in enumerate(results_onnx):
        bbox = person['bbox']
        score = person.get('score', 'N/A')
        print(f"  行人 {i+1}: bbox={bbox}, score={score:.4f}")
    
    # 比较结果
    print("\n=== 结果比较 ===")
    if len(results_orig) == len(results_onnx):
        print("✓ 检测到的人数一致")
    else:
        print(f"✗ 检测到的人数不一致: 原版={len(results_orig)}, ONNX={len(results_onnx)}")
    
    # 如果都检测到了人，比较边界框的相似性
    if len(results_orig) > 0 and len(results_onnx) > 0:
        for i in range(min(len(results_orig), len(results_onnx))):
            bbox1 = results_orig[i]['bbox']
            bbox2 = results_onnx[i]['bbox']
            
            # 计算IoU
            iou = calculate_iou(bbox1, bbox2)
            print(f"  边界框 {i+1} IoU: {iou:.4f}")
            
            if iou > 0.5:
                print(f"    ✓ 边界框相似度较高")
            else:
                print(f"    ✗ 边界框差异较大")

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_u = area_1 + area_2 - area_i
    
    return area_i / area_u if area_u > 0 else 0.0

if __name__ == "__main__":
    compare_detectors()
