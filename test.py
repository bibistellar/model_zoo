from pickletools import read_unicodestring8
import time

import test
from facelib.face_analysis import FaceModel
from bodylib.body_analysis import BodyModel
import numpy as np
import cv2
def test_getfaces():
    try:
        # Create an instance of FaceModel
        face_model = FaceModel()
        face_model.init()  # Initialize the model
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/image1.png")
        if img is None:
            print("Error: Could not read the image.")
            return
        dets, kpss = face_model.getfaces(img)

        if dets is None or len(dets) == 0:
            print("未检测到人脸")
            read_unicodestring8

        # 可视化检测结果
        for i, det in enumerate(dets):
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加置信度文本
            text = f"Conf: {score:.2f}"
            cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 绘制关键点
            if kpss is not None and i < len(kpss):
                kps = kpss[i]
                for j in range(5):
                    pt = kps[j]
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), 2)
        cv2.imwrite("./detected_face234.png",img)
        # print(f"检测结果已保存")
    except Exception as e:
           print(f"Error in test_getfaces: {e}")

def test_getfeat_norm():
    try:
        # Create an instance of FaceModel
        face_model = FaceModel()
        face_model.init()  # Initialize the model
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/image1.png")
        if img is None:
            print("Error: Could not read the image.")
            return
        
        # Get faces from the image
        dets, kpss = face_model.getfaces(img)
        
        if dets is None or len(dets) == 0:
            print("未检测到人脸")
            return
        
        # Get features for the first detected face
        feat_norm = face_model.getfeat_norm(img, kpss[0])
        
        if feat_norm is not None:
            print(f"Feature vector: {feat_norm}")
        else:
            print("未能获取人脸特征向量")
    except Exception as e:
           print(f"Error in test_getfeat_norm: {e}")

def test_ga():
    try:
        # Create an instance of FaceModel
        face_model = FaceModel()
        face_model.init()  # Initialize the model
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/image1.png")
        if img is None:
            print("Error: Could not read the image.")
            return
        
        # Get faces from the image
        dets, kpss = face_model.getfaces(img)
        
        if dets is None or len(dets) == 0:
            print("未检测到人脸")
            return
        
        # 处理每个检测到的人脸
        for i, det in enumerate(dets):
            x1, y1, x2, y2, score = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制人脸边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 检测性别年龄
            gender,age = face_model.get_ga(img, (x1, y1, x2, y2))
            # 在图像上标注性别和年龄
            gender_str = "male" if gender == 1 else "female"
            label = f"{gender_str}, {age}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite("genderage_result.png", img)
    except Exception as e:
           print(f"Error in test_getfeat_norm: {e}")

def test_getresult():
    try:
        # Create an instance of BodyModel
        body_model = BodyModel()
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/19.jpg")
        if img is None:
            print("Error: Could not read the image.")
            return
        
        # Get body detection results
        boxes, attrs = body_model.getresult(img)
        
        print(boxes, attrs)
    except Exception as e:
           print(f"Error in test_getresult: {e}")

def test_getcarresult():
    try:
        # Create an instance of BodyModel
        body_model = BodyModel()
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/19.jpg")
        if img is None:
            print("Error: Could not read the image.")
            return
        
        # Get car detection results
        boxes, vehicleplate, vehicle_attr = body_model.getcarresult(img)
        
        print(boxes, vehicleplate, vehicle_attr)
    except Exception as e:
           print(f"Error in test_getcarresult: {e}")

def test_getBodyEmbeddings():
    try:
        # Create an instance of BodyModel
        body_model = BodyModel()
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/19.jpg")
        if img is None:
            print("Error: Could not read the image.")
            return
        
        # First get body detection results to crop person regions
        boxes, attrs = body_model.getresult(img)
        
        if boxes is None or len(boxes) == 0:
            print("未检测到人体")
            return
        
        print(f"检测到 {len(boxes)} 个人体")
        
        # Process each detected person
        for i, box in enumerate(boxes):
            try:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Crop the person region
                person_crop = img[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    print(f"第 {i+1} 个人体裁剪区域为空，跳过")
                    continue
                
                # Get ReID embeddings for this person
                embeddings = body_model.getBodyEmbeddings(person_crop)
                
                if embeddings is not None:
                    print(f"第 {i+1} 个人体特征向量维度: {embeddings.shape}")
                    print(f"特征向量范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
                    print(f"特征向量L2范数: {np.linalg.norm(embeddings):.4f}")
                    
                    # Save the cropped person image for reference
                    cv2.imwrite(f"person_crop_{i+1}.jpg", person_crop)
                    print(f"已保存人体裁剪图像: person_crop_{i+1}.jpg")
                else:
                    print(f"第 {i+1} 个人体特征提取失败")
                    
            except Exception as e:
                print(f"处理第 {i+1} 个人体时出错: {e}")
        
        print("ReID特征提取测试完成")
        
    except Exception as e:
        print(f"Error in test_getBodyEmbeddings: {e}")

if __name__ == "__main__":
    # test_getfaces()
    # test_getfeat_norm()
    # test_ga()
    # test_getresult()
    # test_getcarresult()
    test_getBodyEmbeddings()