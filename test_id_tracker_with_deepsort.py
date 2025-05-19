import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from people_extractor import PeopleExtractor
from feature_extractor import FeatureExtractor

# 初始化 YOLOv5 模型（可以替换为 yolov5s、yolov5m 等）
people_extractor = PeopleExtractor()
feature_extractor = FeatureExtractor()

# 初始化 Deep SORT
tracker = DeepSort(max_age=30, n_init=10, nms_max_overlap=0.5, max_cosine_distance=0.15)

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

    # YOLOv5 目标检测（只检测人类）
    detections = people_extractor.extract_boxes(frame)
    # make images
    person_images = [frame[int(y1):int(y2), int(x1):int(x2)] for x1,y1,x2,y2, conf, cls in detections]
    batch_features = [feature_extractor.extract(person_image).flatten() for person_image in person_images]

    # 提取所有人的边界框和置信度
    dets_for_tracker = []
    for *box, conf, cls in detections:
        if int(cls) == 0:  # class 0 is 'person'
            x1, y1, x2, y2 = map(int, box)
            dets_for_tracker.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

    # 更新 tracker
    tracks = tracker.update_tracks(dets_for_tracker, frame=frame,embeds=batch_features)

    # 可视化
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
