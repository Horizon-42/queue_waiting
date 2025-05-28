import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from people_extractor import PeopleExtractor
from feature_extractor import FeatureExtractor
from thread_safe_ordered_dict import ThreadSafeOrderedDict,TrackInfo, Person
import numpy as np

class IDTracker:
    def __init__(self, camera_id:int, global_tracks:ThreadSafeOrderedDict):
        self.camera_id = camera_id
        self.people_extractor = PeopleExtractor()
        self.feature_extractor = FeatureExtractor()
        # 初始化 Deep SORT
        self.tracker = DeepSort(max_age=30, n_init=5, nms_max_overlap=0.7)

        self.tracked_infos:dict[int, TrackInfo] = {}

        self.current_frame = 0

        self.global_tracks = global_tracks 

    def process_frame(self, frame):
        self.current_frame += 1
        # YOLOv5 目标检测（只检测人类）
        detections, poses = self.people_extractor.extract_boxes_with_poses(frame)
        frame = self.people_extractor.draw_poses(frame, poses)

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
            track_id = track.track_id
            if track_id not in self.tracked_infos:
                self.tracked_infos[track_id] = TrackInfo(id=-1, view_id=self.camera_id, 
                                                         in_view_time=self.current_frame, 
                                                         out_view_time=self.current_frame, feature=track.features[0])
            else:
                track_info = self.tracked_infos[track_id]
                track_info.out_view_time = self.current_frame
                duration = track_info.out_view_time - track_info.in_view_time
                # update the newest feature
                track_info.feature = track_info.feature + 1/duration * (track.features[0] - track_info.feature)
                # track_info.feature = track.features[0] 
            self.__compare_across_views(self.tracked_infos[track_id])
            l, t, r, b = track.to_ltrb()
            track_info = self.tracked_infos[track_id]
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"{track_info.id} at view {self.camera_id} : {track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # show people count
        cv2.putText(frame, f"Total: {len(self.global_tracks.keys())}, In view: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, tracks
    
    def cosine_similarity(self, feature1, feature2):
        if feature1 is None or feature2 is None:
            return 0
        return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
    
    def __compare_across_views(self, track_info:TrackInfo):
        THRESHOLD = 0.4
        min_dist = 1
        min_id = -1
        # TODO 最大池化
        for person in self.global_tracks.values():
            min_view_dist = 1
            for view_id, feature in person.view_features.items():
                if view_id == track_info.view_id:
                    continue
                # compare the feature with the global tracks
                dist = 1- self.cosine_similarity(track_info.feature, feature)
                min_view_dist = min(min_view_dist, dist)
            if min_view_dist < min_dist:
                min_dist = min_view_dist
                min_id = person.id
        print(min_dist)
        if min_dist < THRESHOLD:
            # update the global track
            track_info.id = min_id
            self.global_tracks.update(min_id, track_info.view_id, track_info.feature)
        else:
            # add a new global track
            self.global_tracks.add(track_info.view_id, track_info.feature)
            track_info.id = self.global_tracks.next_id - 1
            


