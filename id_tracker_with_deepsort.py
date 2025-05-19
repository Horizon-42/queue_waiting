import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from people_extractor import PeopleExtractor
from feature_extractor import FeatureExtractor


class IDTracker:
    def __init__(self, camera_id:int):
        self.camera_id = camera_id
        self.people_extractor = PeopleExtractor()
        self.feature_extractor = FeatureExtractor()
        # 初始化 Deep SORT
        self.tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)