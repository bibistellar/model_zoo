import os
import sys
import cv2
import numpy as np
import ray
import time

# 确保能够导入上一级目录中的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def test_genderage():
    # 连接到Ray集群
    ray.init(address="auto", ignore_reinit_error=True)
    
    try:
        # 获取已部署的GenderAge Actor
        genderage_actor = ray.get_actor("GenderAge", namespace="face_analysis")
        print("成功连接到GenderAge Actor")
        
        # 首先需要人脸检测
        try:
            face_detector = ray.get_actor("Buffalo", namespace="face_detection")
            print("成功连接到Buffalo Actor进行人脸检测")
        except:
            print("警告：无法连接到Buffalo Actor，请确保先运行了buffalo_l目录下的ray_buffalo_l_onnx.py")
            return
        
        # 加载测试图像（使用buffalo_l目录下的示例图像）
        test_img = os.path.join(parent_dir, "buffalo_l", "image1.png")
        if not os.path.exists(test_img):
            print(f"测试图像不存在: {test_img}")
            return
            
        img = cv2.imread(test_img)
        if img is None:
            print(f"无法读取图像: {test_img}")
            return
            
        print(f"加载测试图像: {test_img}")
        
        # 检测人脸
        dets_ref = ray.get(face_detector.detect.remote(img))
        dets, kpss = dets_ref
        
        if dets is None or len(dets) == 0:
            print("未检测到人脸")
            return
        
        print(f"检测到 {len(dets)} 张人脸")
        
        # 处理每个检测到的人脸
        result_img = img.copy()
        for i, det in enumerate(dets):
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制人脸边界框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 检测性别年龄
            start_time = time.time()
            gender_str, age, _ = ray.get(genderage_actor.detect_genderage.remote(img, (x1, y1, x2, y2)))
            end_time = time.time()
            
            if gender_str == "男":
                gender_str = "male"
            else:
                gender_str = "female"

            if gender_str is not None and age is not None:
                # 添加性别和年龄标签
                text = f"{gender_str}, {age} years old (score: {score:.2f})"
                cv2.putText(result_img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                print(f"face #{i+1}: {gender_str}, {age} years old, process time: {(end_time-start_time)*1000:.2f}ms")
        
        # 保存结果图像
        output_file = os.path.join(current_dir, "genderage_result.png")
        cv2.imwrite(output_file, result_img)
        print(f"结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    test_genderage()
