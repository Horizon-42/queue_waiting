import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from people_extractor import PeopleExtractor
from feature_extractor import FeatureExtractor
from id_tracker_with_deepsort import IDTracker

# 初始化 YOLOv5 模型（可以替换为 yolov5s、yolov5m 等）
tracker = IDTracker(camera_id=0)
# 打开视频文件
video_path = "dataset/mid.mp4"
cap = cv2.VideoCapture(video_path)

# 设置输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_tracked.mp4", fourcc, 30.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracked_frame, tracks = tracker.process_frame(frame)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
