from pickletools import read_unicodestring8
from facelib.face_analysis import FaceModel
import numpy as np
import cv2
def test_getfaces():
    try:
        # Create an instance of FaceModel
        face_model = FaceModel()
        face_model.init()  # Initialize the model
        
        # Load a test image
        img = cv2.imread("/root/Code/aiplat/model_zoo/facelib/image1.png")
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
        img = cv2.imread("/root/Code/aiplat/model_zoo/facelib/image1.png")
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

if __name__ == "__main__":
    test_getfaces()
    test_getfeat_norm()