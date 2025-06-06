import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path='yolov11n.pt', conf_thresh=0.5):
        self.model = YOLO("yolo11n.pt")
        self.conf_thresh = conf_thresh

    def detect_and_crop(self, frame):
        """
        对图像进行人体检测并裁剪出所有人体区域。

        :param frame: BGR格式图像（numpy数组）
        :return: List[dict] - 每个元素为 {'crop': ndarray, 'bbox': [x1, y1, x2, y2]}
        """
        results = self.model(frame)[0]
        crops = []
        for box in results.boxes:
            if int(box.cls) != 0 or box.conf < self.conf_thresh:
                continue  # 只保留类别为 "person" 的检测框
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append({'crop': crop, 'bbox': [x1, y1, x2, y2]})
        return crops

# 示例用法
if __name__ == "__main__":
    import os

    # 加载图像（替换为你自己的图片路径）
    image_path = "/root/Code/aiplat/model_zoo/image1.png"
    if not os.path.exists(image_path):
        print(f"图像 {image_path} 不存在")
        exit(1)

    frame = cv2.imread(image_path)
    detector = PersonDetector()

    results = detector.detect_and_crop(frame)

    print(f"检测到 {len(results)} 个行人")
    for i, person in enumerate(results):
        crop = person['crop']
        x1, y1, x2, y2 = person['bbox']
        cv2.imshow(f"Crop_{i}", crop)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Original with Boxes", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()