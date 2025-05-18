from feature_extractor import FeatureExtractor
from people_extractor import PeopleExtractor
import cv2
import torch
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
from fastreid.evaluation.rerank import re_ranking

@dataclass
class TrackedObject:
    camera_id: int
    id: int
    bbox: list
    last_seen: int
    features: torch.Tensor
    features_buffer:list[torch.Tensor] = None


class IDTracker:
    def __init__(self, camera_id:int):
        self.camera_id = camera_id
        self.people_extractor = PeopleExtractor()
        self.feature_extractor = FeatureExtractor()
        self.tracked_objects:dict[int, TrackedObject] = {}  # Dictionary to store tracked objects
        self.next_id = 0  # Next ID to assign to a new object
        self.now_seen = 0  # Counter for the number of frames seen

    def track(self, frame):
        boxes, imgs = self.people_extractor.extract(frame)
        for box,person_image in zip(boxes, imgs):
            features = self.feature_extractor.extract(person_image)
            # Normalize the features
            features = F.normalize(features, p=2, dim=1)

            matched_id = self.match(features, box)
            if matched_id is not None:
                self.update_object(matched_id, box, features)
            else:
                # Assign a new ID to the new object
                self.create_new_object(box, features)
        self.now_seen += 1
        return self.tracked_objects
    
    def create_new_object(self, box, features):
        # Assign a new ID to the new object
        self.tracked_objects[self.next_id] = TrackedObject(
            camera_id=self.camera_id,
            id=self.next_id,
            bbox=box,
            last_seen=self.now_seen,
            features=features,
            features_buffer=[features]
        )
        self.next_id += 1
    
    def update_object(self, obj_id, box, features):
        # Update the existing tracked object
        self.tracked_objects[obj_id].bbox = box
        self.tracked_objects[obj_id].last_seen = self.now_seen
        self.tracked_objects[obj_id].features = features
        # self.tracked_objects[obj_id].features_buffer.append(features)
        # if len(self.tracked_objects[obj_id].features_buffer) > 5:
        #     self.tracked_objects[obj_id].features_buffer.pop(0)
        #     self.tracked_objects[obj_id].features = torch.mean(torch.stack(self.tracked_objects[obj_id].features_buffer), dim=0)
    
    def match_list(self, frame_features, boxes):
        """
        Match all detection features in current frame with tracked objects.

        Args:
            frame_features (Tensor): shape (N, D), N is the number of detected objects

        Returns:
            List[int]: matching object IDs, -1 means no match
        """
        matched_ids = []

        # 如果没有历史目标，全部返回 -1
        if len(self.tracked_objects) == 0:
            return [-1 for _ in range(frame_features.shape[0])]

        # 收集历史特征
        past_features = []
        obj_ids = []

        for obj_id, obj in self.tracked_objects.items():
            past_features.append(obj.features)  # shape (1, D)
            obj_ids.append(obj_id)

        past_features = torch.cat(past_features, dim=0)  # (M, D)
        # make sure it is M,D
        # if len(past_features.shape) == 1:
        past_features = past_features.reshape(len(obj_ids), -1)  # (M, D)


        # 转为 numpy
        q_feat = frame_features.cpu().numpy()      # (N, D)
        g_feat = past_features.cpu().numpy()       # (M, D)

        # 初始距离矩阵（cosine distance）
        q_g_dist = 1 - np.matmul(q_feat, g_feat.T)     # (N, M)
        # q_q_dist = 1 - np.matmul(q_feat, q_feat.T)     # (N, N)
        # g_g_dist = 1 - np.matmul(g_feat, g_feat.T)     # (M, M)

        # # re-ranking
        # reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)  # (N, M)
        reranked_dist = q_g_dist.copy()  # (N, M)

        DIST_THRESHOLD = 0.2 # 可调参数：距离小于该阈值才认为是匹配

        for i in range(reranked_dist.shape[0]):
            dist_row = reranked_dist[i]  # 当前特征与所有历史目标的距离
            min_idx = np.argmin(dist_row)
            min_dist = dist_row[min_idx]

            if min_dist <= DIST_THRESHOLD:
                obj_id = obj_ids[min_idx]
                # 判断bbox是否重叠
                pre_bbox = self.tracked_objects[obj_id].bbox
                now_bbox = boxes[i]
                # 计算重叠面积
                x1 = max(pre_bbox[0], boxes[i][0])
                y1 = max(pre_bbox[1], boxes[i][1])
                x2 = min(pre_bbox[2], boxes[i][2])
                y2 = min(pre_bbox[3], boxes[i][3])
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                overlap_area = w * h
                # 计算当前bbox的面积
                current_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                # 计算重叠率
                overlap_rate = overlap_area / current_area
                if overlap_rate > 0.8:
                    matched_ids.append(obj_id)
                else:
                    matched_ids.append(-1)
            else:
                matched_ids.append(-1)

        return matched_ids

    def box_iou(self, box1, box2):
        # Calculate the intersection over union (IoU) of two bounding boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        intersection_area = w * h
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def match(self, features, bbox):
        # Compare the features with the tracked objects
        HARD_THRESH = 0.3

        past_features = []
        obj_ids = []
        for obj_id, obj in self.tracked_objects.items():
            past_features.append(obj.features)
            obj_ids.append(obj_id)
        if len(past_features) == 0:
            return None
        past_features = torch.cat(past_features, dim=0) 

        dists = [1-F.cosine_similarity(features, past_feature)[0] for past_feature in past_features]

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        # print(f"min_dist: {min_dist.item()}, min_idx: {min_idx.item()}")
        
        if min_dist.item() < HARD_THRESH:
            return obj_ids[min_idx.item()]
        return None

    def compare_features(self, features1, features2):
        # can use cosine similarity or Euclidean distance
        cosine_similarity = torch.nn.functional.cosine_similarity(features1, features2)
        return cosine_similarity > 0.6
    def draw_boxes(self, frame):
        for obj_id, obj in self.tracked_objects.items():
            if obj.last_seen < self.now_seen - 5:
                continue
            bbox = obj.bbox
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
