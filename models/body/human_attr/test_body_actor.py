#!/usr/bin/env python3
"""
Body Detection Ray Actor 测试脚本
用于验证paddle依赖隔离是否正常工作
"""

import cv2
import ray
import sys
import os

def test_body_actor_deployment():
    """测试Body Actor的部署和运行"""
    
    # 连接到Ray集群
    try:
        ray.init(address="auto", ignore_reinit_error=True)
        print("✓ 成功连接到Ray集群")
    except Exception as e:
        print(f"✗ 连接Ray集群失败: {e}")
        return False
    
    # 尝试获取Body Actor
    try:
        body_actor = ray.get_actor("bodydetect", namespace="body_detection")
        print("✓ 成功获取Body Actor")
    except Exception as e:
        print(f"✗ 获取Body Actor失败: {e}")
        print("请先运行 python deploy_ray.py 部署Actor")
        return False
    
    # 健康检查
    try:
        health_status = ray.get(body_actor.health_check.remote())
        print(f"✓ Actor健康检查: {health_status}")
        
        if health_status.get("status") == "healthy":
            print("✓ Body Actor运行在正确的paddle环境中")
            return True
        else:
            print("✗ Body Actor环境配置有问题")
            return False
            
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False

def test_body_detection():
    """测试身体检测功能"""
    try:
        body_actor = ray.get_actor("bodydetect", namespace="body_detection")
        
        # 使用简单的测试图像（这里需要一个真实的图像路径）
        img_path ='/root/Code/aiplat/model_zoo/19.jpg'
        img = cv2.imread(img_path)
        bboxes, attrs_list = ray.get(body_actor.getresult.remote(img))
        print(f"检测结果: {bboxes, attrs_list}")
        
        print("✓ Body Actor可以正常调用（实际检测需要提供图像）")
        return True
        
    except Exception as e:
        print(f"✗ 身体检测测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("Body Detection Ray Actor 测试")
    print("=" * 50)
    
    # 测试部署
    deployment_ok = test_body_actor_deployment()
    
    if deployment_ok:
        # 测试功能
        test_body_detection()
    
    print("\n测试完成")

if __name__ == "__main__":
    main()
