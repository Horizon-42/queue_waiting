import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown
from mmpose.apis.inference_tracking import _compute_iou
from mmpose.utils import register_all_modules
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化mmpose模型
register_all_modules()
config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
pose_model = init_model(config_file, checkpoint_file, device='cuda:0')

# 初始化DeepSORT
tracker = DeepSort(max_age=30)  # 参数可以调节

# 读取视频
video_path = 'dataset/end.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MMpose推理，得到每个人的关键点和bbox
    results = inference_topdown(pose_model, frame)

    bboxes = []
    for res in results:
        bbox = res.pred_instances.bboxes[0]
        bboxes.append(bbox)

    # DeepSORT追踪
    # 输入格式是 [[x1,y1,x2,y2], ...]
    tracks = tracker.update_tracks(bboxes, frame=frame)

    # 画框和ID，并绘制关键点
    for track in tracks:
        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)

        # 找到对应track_id的关键点
        # 简单方法：通过bbox重合或距离匹配results
        matched_kpts = None
        for res in results:
            bx = res.pred_instances.bboxes[0]
            iou = _compute_iou(bx, ltrb)
            if iou > 0.5:
                matched_kpts = res['keypoints']
                break

        # 画跟踪框和ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 画关键点
        if matched_kpts is not None:
            for x, y, conf in matched_kpts:
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    cv2.imshow('MMpose + DeepSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
        break

cap.release()
cv2.destroyAllWindows()

# # 计算IOU函数
# def compute_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou
# 