from deep_sort_realtime.deepsort_tracker import DeepSort,Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.deep_sort import nn_matching
from feature_extractor import FeatureExtractor
from people_extractor import PeopleExtractor
import cv2

class SingleViewTracker:
    def __init__(self, max_cosine_distance=0.2, nn_budget=None):
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        # Initialize the tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric=metric, max_age=100)

        # Initialize the list of tracked objects
        self.tracked_objects = {}
        self.next_id = 0  # Counter for the next ID to assign

        self.people_extractor = PeopleExtractor()
        self.feature_extractor = FeatureExtractor()

    def track(self, frame):
        # Extract peoples from the frame
        boxes, imgs = self.people_extractor.extract(frame)

        features = []
        for person_image in imgs:
            feature = self.feature_extractor.extract(person_image).cpu().numpy().flatten()
            features.append(feature)
        

        # Create detections
        # q: what is detection?
        # A detection is a bounding box with an associated feature vector
        detections = []
        # for box, feature in zip(boxes, features):
        for box in boxes:
            x1,y1,x2,y2 = box
            ltwh = (x1, y1, x2 - x1, y2 - y1)
            detection = Detection(ltwh=ltwh, confidence=1.0,feature=feature)
            detections.append(detection)

        # Update the tracker with the new detections
        self.tracker.predict()
        self.tracker.update(detections)
        # Update the tracked objects
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            self.tracked_objects[track_id] = {
                'bbox': bbox,
                'track_id': track_id,
                'last_seen': track.time_since_update,
            }
        # Return the tracked objects
        return self.tracked_objects
    
    # visualize the tracked objects
    def visualize(self, frame):
        for track_id, obj in self.tracked_objects.items():
            x1,y1,x2,y2 = map(int, obj['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    

if __name__ == "__main__":
    video_path = 'dataset/end.mp4'
    cap = cv2.VideoCapture(video_path)
    tracker = SingleViewTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tracked_objects = tracker.track(frame)
        frame = tracker.visualize(frame)

        cv2.imshow('Tracked Objects', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    
