import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from people_extractor import PeopleExtractor
from feature_extractor import FeatureExtractor
from thread_safe_ordered_dict import WaitingQueue

class IDTracker:
    def __init__(self, camera_id:int):
        self.camera_id = camera_id
        self.people_extractor = PeopleExtractor()
        self.feature_extractor = FeatureExtractor()
        # 初始化 Deep SORT
        self.tracker = DeepSort(max_age=60, n_init=3, nms_max_overlap=1.0)
    
    def process_frame(self, frame):
        # YOLOv5 目标检测（只检测人类）
        detections, poses = self.people_extractor.extract_boxes_with_poses(frame)
        # make images
        person_images = [frame[int(y1):int(y2), int(x1):int(x2)] for x1,y1,x2,y2, conf, cls in detections]
        batch_features = [self.feature_extractor.extract(person_image).flatten() for person_image in person_images]

        # 提取所有人的边界框和置信度
        dets_for_tracker = []
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            dets_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))
        
        tracks = self.tracker.update_tracks(dets_for_tracker, frame=frame, embeds=batch_features)

        for track in tracks:
            if not track.is_confirmed():
                continue
            


