import cv2
import numpy as np
import onnxruntime as ort

class PersonDetectorONNX:
    def __init__(self, model_path='/root/Code/aiplat/model_zoo/reid/yolo11n.onnx', conf_thresh=0.5, iou_thresh=0.45):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_shape = (640, 640)  # 假设输入固定大小，如需动态适配需解析模型输入 shape

    def preprocess(self, image):
        # 保存原始尺寸
        self.orig_h, self.orig_w = image.shape[:2]
        
        # 计算缩放比例，保持宽高比
        scale = min(self.input_shape[0] / self.orig_w, self.input_shape[1] / self.orig_h)
        new_w = int(self.orig_w * scale)
        new_h = int(self.orig_h * scale)
        
        # 缩放图像
        img = cv2.resize(image, (new_w, new_h))
        
        # 创建填充后的图像
        padded_img = np.ones((self.input_shape[1], self.input_shape[0], 3), dtype=np.uint8) * 114
        
        # 计算填充位置
        dw = (self.input_shape[0] - new_w) // 2
        dh = (self.input_shape[1] - new_h) // 2
        
        # 将缩放后的图像放到填充图像的中心
        padded_img[dh:dh+new_h, dw:dw+new_w] = img
        
        # 保存缩放和填充信息用于后处理
        self.scale = scale
        self.dw = dw
        self.dh = dh
        
        # BGR -> RGB -> CHW
        img = padded_img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img[np.newaxis, :]  # NCHW

    def postprocess(self, outputs, orig_shape):
        predictions = outputs[0]

        # squeeze batch 维度
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)  # (84, 8400)

        # 检查是否需要转置，YOLO输出通常是 (classes+coords, detections)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T  # (8400, 84)

        # 分离坐标、置信度和类别预测
        # YOLO11格式: [x, y, w, h, conf1, conf2, ..., conf80]
        # 对于80类COCO数据集，person是第0类
        boxes = []
        
        for det in predictions:
            if len(det) < 5:
                continue
                
            x, y, w, h = det[0:4]
            
            # 对于YOLO11，直接使用类别置信度
            if len(det) >= 84:  # 4 coords + 80 classes
                cls_scores = det[4:]
                cls_id = np.argmax(cls_scores)
                score = cls_scores[cls_id]
            else:
                continue
            
            # 只保留person类 (类别ID为0) 且置信度足够的检测
            if cls_id != 0 or score < self.conf_thresh:
                continue
            
            # 转换为角点格式并映射回原图坐标
            x1 = (x - w / 2 - self.dw) / self.scale
            y1 = (y - h / 2 - self.dh) / self.scale
            x2 = (x + w / 2 - self.dw) / self.scale
            y2 = (y + h / 2 - self.dh) / self.scale
            
            # 确保坐标在图像范围内
            x1 = max(0, min(self.orig_w, x1))
            y1 = max(0, min(self.orig_h, y1))
            x2 = max(0, min(self.orig_w, x2))
            y2 = max(0, min(self.orig_h, y2))
            
            # 检查box是否有效
            if x2 > x1 and y2 > y1:
                boxes.append([int(x1), int(y1), int(x2), int(y2), score])
            
        return self.non_max_suppression(boxes, self.iou_thresh)

    def non_max_suppression(self, boxes, iou_threshold):
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        x1 = boxes[:, 0]; y1 = boxes[:, 1]
        x2 = boxes[:, 2]; y2 = boxes[:, 3]
        scores = boxes[:, 4]

        indices = np.argsort(scores)[::-1]
        keep = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            union = (x2[i]-x1[i]) * (y2[i]-y1[i]) + (x2[indices[1:]]-x1[indices[1:]]) * (y2[indices[1:]]-y1[indices[1:]]) - inter
            iou = inter / (union + 1e-6)
            indices = indices[1:][iou <= iou_threshold]
        return boxes[keep].tolist()

    def detect_and_crop(self, frame):
        # 运行ONNX推理
        outputs = self.session.run(None, {'images': self.preprocess(frame)})
        
        # 后处理得到检测框
        boxes = self.postprocess(outputs, frame.shape)
        
        crops = []
        for box in boxes:
            x1, y1, x2, y2, score = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 确保是整数
            h, w = frame.shape[:2]
            
            # 确保坐标在图像范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = frame[y1:y2, x1:x2]
            crops.append({'crop': crop, 'bbox': [x1, y1, x2, y2], 'score': score})
            
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
    detector = PersonDetectorONNX()

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