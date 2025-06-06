import cv2
import numpy as np
import ray

import runtime_config as runtime_config

class BodyModel:
    def __init__(self):
        #检测是否已连接ray集群，没有则连接，有则跳过
        ray.init(address=runtime_config.ray_address,ignore_reinit_error=True)  # 连接到集群

    def getresult(self, img):
        body_actor = ray.get_actor("bodydetect", namespace="body_detection")
        bboxes, attrs_list = ray.get(body_actor.getresult.remote(img))
        return bboxes, attrs_list
    
    def getcarresult(self, img):
        body_actor = ray.get_actor("bodydetect", namespace="body_detection")
        boxes, vehicleplate, vehicle_attr = ray.get(body_actor.getcarresult.remote(img))
        return boxes, vehicleplate, vehicle_attr
    
    def getBodyEmbeddings(self, img):
        reid_actor = ray.get_actor(runtime_config.reid_config["name"], namespace=runtime_config.reid_config["namespace"])
        embeddings = ray.get(reid_actor.extract_feature.remote(img))
        return embeddings