from pdb import run
import platform
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ray
from ray.runtime_env import RuntimeEnv
import logging
import os
import onnxruntime as ort
from runtime_config import conda_env
ort.set_default_logger_severity(3)

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    pts = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i+1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        pts.extend([px, py])
    return np.stack(pts, axis=-1)

# 初始化Ray - 如果在本地运行，可以直接初始化
# 如果已经有集群，则连接到集群
@ray.remote(num_cpus=1, num_gpus=1, runtime_env=conda_env)
class BuffaloFaceDetector:
    """Ray Actor 封装RetinaFace人脸检测功能"""
    
    def __init__(self, model_file=None, input_size=(640, 640), nms_thresh=0.4, det_thresh=0.5):
        """初始化RetinaFace检测器"""
        # 默认查找当前目录下的模型文件
        if model_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_file = os.path.join(current_dir, "det_10g.onnx")
            if not os.path.exists(model_file):
                print(f"警告: 默认模型文件不存在: {model_file}")
                print("请提供有效的模型文件路径")
                return
        
        # 初始化RetinaFace检测器
        providers = ['CPUExecutionProvider']
        # 设置ONNX Runtime会话选项，禁用输出形状检查
        sess_options = ort.SessionOptions()
        sess_options.add_session_config_entry("session.check_shape_compatibility", "False")
        sess_options.add_session_config_entry("session.ignore_dimension_mismatch", "True")
        
        try:
            self.session = ort.InferenceSession(
                model_file, 
                sess_options=sess_options,
                providers=providers
            )
            self.nms_thresh = nms_thresh
            self.det_thresh = det_thresh
            self.user_input_size = input_size
            self._init_vars()
            print(f"已加载RetinaFace模型: {model_file}")
            print(f"输入尺寸: {self.input_size}, NMS阈值: {self.nms_thresh}, 检测阈值: {self.det_thresh}")
        except Exception as e:
            print(f"加载模型时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _init_vars(self):
        try:
            inp = self.session.get_inputs()[0]
            name = inp.name
            shape = inp.shape  # [batch, channel, height, width]

            if self.user_input_size:
                # 直接使用用户指定的输入尺寸，避免形状推断引起的问题
                self.input_size = self.user_input_size
            else:
                h, w = shape[2], shape[3]
                if isinstance(h, int) and isinstance(w, int):
                    self.input_size = (w, h)
                else:
                    # 如果形状是动态的，使用默认值
                    self.input_size = (640, 640)
                    print(f"警告: 检测到动态形状 {shape}，使用默认输入尺寸 {self.input_size}")

            self.input_name = name
            self.output_names = [o.name for o in self.session.get_outputs()]
            out_count = len(self.output_names)
            self.use_kps = out_count in (9, 15)
            if out_count in (6, 9):
                self.fmc = 3; self._feat_stride_fpn = [8, 16, 32]; self._num_anchors = 2
            else:
                self.fmc = 5; self._feat_stride_fpn = [8, 16, 32, 64, 128]; self._num_anchors = 1
        except Exception as e:
            print(f"初始化变量时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def forward(self, img, threshold):
        try:
            # 确保输入图像尺寸与模型要求一致
            # 首先调整图像大小以确保一致性
            resized_img = cv2.resize(img, self.input_size)
            
            # 创建正确大小的blob
            blob = cv2.dnn.blobFromImage(
                resized_img, 1/128.0, self.input_size, (127.5,127.5,127.5), swapRB=True
            )
            
            # 运行推理
            outs = self.session.run(self.output_names, {self.input_name: blob.astype(np.float32)})

            h_in, w_in = blob.shape[2:]
            scores_list, bbox_list, kps_list = [], [], []
            for idx, stride in enumerate(self._feat_stride_fpn):
                if idx >= len(outs) or idx+self.fmc >= len(outs):
                    print(f"警告: 输出数量不足，跳过stride {stride}")
                    continue
                    
                try:
                    scores = outs[idx].ravel()
                    bpred = outs[idx+self.fmc].reshape(-1,4) * stride
                    if self.use_kps and idx+self.fmc*2 < len(outs):
                        kpred = outs[idx+self.fmc*2].reshape(-1,10) * stride
                    else:
                        # 如果没有关键点输出，设置一个空数组
                        kpred = np.array([])

                    h = h_in // stride; w = w_in // stride
                    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
                    centers = np.stack([xv, yv],-1).reshape(-1,2).astype(np.float32)*stride
                    if self._num_anchors>1:
                        centers = np.repeat(centers, self._num_anchors, axis=0)

                    idxs = scores >= threshold
                    scores_list.append(scores[idxs])
                    bbox_list.append(distance2bbox(centers, bpred)[idxs])
                    if self.use_kps and kpred.size > 0:
                        kps = distance2kps(centers, kpred)
                        kps_list.append(kps[idxs].reshape(-1,5,2))
                except Exception as e:
                    print(f"处理stride {stride}时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            return scores_list, bbox_list, kps_list
        except Exception as e:
            print(f"前向推理时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def nms(self, dets):
        x1,y1,x2,y2,s = dets.T
        areas = (x2-x1+1)*(y2-y1+1)
        order = s.argsort()[::-1]
        keep=[]
        while order.size:
            i=order[0]; keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w=np.maximum(0,xx2-xx1+1); h=np.maximum(0,yy2-yy1+1)
            inter=w*h; ovr=inter/(areas[i]+areas[order[1:]]-inter)
            order=order[np.where(ovr<=self.nms_thresh)[0]+1]
        return keep

    def detect(self, img, max_num=0):
        """
        检测图像中的人脸
        
        Args:
            img: 输入图像(BGR格式)
            max_num: 最大返回人脸数量，0表示全部返回
            
        Returns:
            det: 检测结果 [x1, y1, x2, y2, score]
            kpss: 关键点坐标 (如果支持)
        """
        if img is None:
            print("警告: 输入图像为None")
            return None, None
            
        try:
            print("正在检测人脸...")
            # 保存原始图像尺寸，用于后处理
            orig_h, orig_w = img.shape[:2]
            
            # 调整图像大小以适应模型输入，但保持宽高比
            mw, mh = self.input_size
            # 计算缩放比例
            scale_w = mw / orig_w
            scale_h = mh / orig_h
            scale = min(scale_w, scale_h)
            
            # 计算调整后的尺寸
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # 缩放图像
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # 创建目标尺寸的画布并放置调整后的图像
            canvas = np.zeros((mh, mw, 3), dtype=np.uint8)
            # 将调整后的图像放在画布中央
            dx = (mw - new_w) // 2
            dy = (mh - new_h) // 2
            canvas[dy:dy+new_h, dx:dx+new_w] = resized_img

            # 前向传播
            scores, bboxes, kpss = self.forward(canvas, self.det_thresh)
            
            if not scores or len(scores) == 0:
                print("未检测到人脸")
                return None, None
            
            # 将所有结果堆叠
            try:
                scr = np.hstack(scores) if scores else np.array([])
                box = np.vstack(bboxes) if bboxes else np.array([])
                
                if len(scr) == 0 or len(box) == 0:
                    return None, None
                
                # 应用NMS
                ord = np.argsort(scr)[::-1]
                pd = np.hstack((box, scr[:,None])).astype(np.float32)[ord]
                keep = self.nms(pd)
                det = pd[keep]
                
                # 处理关键点
                kprets = None
                if self.use_kps and kpss and len(kpss) > 0:
                    kp = np.vstack(kpss)[ord][keep]
                    kprets = kp
                
                # 调整检测框和关键点坐标到原始图像空间
                if len(det) > 0:
                    # 调整检测框
                    det[:, 0] = (det[:, 0] - dx) / scale  # x1
                    det[:, 1] = (det[:, 1] - dy) / scale  # y1
                    det[:, 2] = (det[:, 2] - dx) / scale  # x2
                    det[:, 3] = (det[:, 3] - dy) / scale  # y2
                    
                    # 裁剪边界框到原始图像大小
                    det[:, 0] = np.clip(det[:, 0], 0, orig_w)
                    det[:, 1] = np.clip(det[:, 1], 0, orig_h)
                    det[:, 2] = np.clip(det[:, 2], 0, orig_w)
                    det[:, 3] = np.clip(det[:, 3], 0, orig_h)
                    
                    # 调整关键点坐标
                    if kprets is not None:
                        for i in range(kprets.shape[0]):
                            for j in range(5):
                                kprets[i, j, 0] = (kprets[i, j, 0] - dx) / scale
                                kprets[i, j, 1] = (kprets[i, j, 1] - dy) / scale
                                # 裁剪关键点到原始图像大小
                                kprets[i, j, 0] = np.clip(kprets[i, j, 0], 0, orig_w)
                                kprets[i, j, 1] = np.clip(kprets[i, j, 1], 0, orig_h)
                
                # 限制返回的人脸数量
                if max_num and det.shape[0] > max_num:
                    print(f"检测到 {det.shape[0]} 张人脸，限制返回前 {max_num} 张")
                    return det[:max_num], kprets[:max_num] if kprets is not None else None
                print(f"检测到 {det.shape[0]} 张人脸")
                return det, kprets
            except Exception as e:
                print(f"处理检测结果时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
        except Exception as e:
            print(f"检测人脸时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    # 添加一个便捷的可视化方法
    def detect_face(self, image, visualize=True):
        """
        检测图片中的人脸并可视化结果
        
        Args:
            image: 图像数据（BGR格式）
            visualize: 是否返回可视化结果
            
        Returns:
            如果visualize为True: 带有检测框和关键点的图像
            如果visualize为False: (检测结果, 关键点坐标)
        """
        if image is None:
            print("警告: 输入图像为None")
            return image if visualize else (None, None)
        
        # 创建图像的可写副本
        img = image.copy()
        
        try:
            # 检测人脸
            dets, kpss = self.detect(img)
            
            if dets is None or len(dets) == 0:
                print("未检测到人脸")
                return img if visualize else (None, None)
            
            if not visualize:
                return dets, kpss
            
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
            
            return img
        except Exception as e:
            print(f"检测人脸时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return img if visualize else (None, None)