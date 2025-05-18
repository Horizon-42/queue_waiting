from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer
import cv2
import os
from tqdm import tqdm
from people_extractor import PeopleExtractor

# 初始化 MMPose 模块
register_all_modules()

# 模型配置与权重
config_file = 'rtmo-l_16xb16-700e_crowdpose-640x640.py'
checkpoint_file = 'rtmo-l_16xb16-700e_crowdpose-640x640-1008211f_20231211.pth'
# config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
# checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

# 加载模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 初始化可视化器
visualizer = PoseLocalVisualizer()
visualizer.set_dataset_meta(model.dataset_meta)

# 输入输出路径
video_path = 'dataset/end.mp4'
output_path = 'output_video.mp4'
out_put_dir = "results"
if not os.path.exists(out_put_dir):
    os.makedirs(out_put_dir)

# 打开视频
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 设置视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'，根据需求
out_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))



# 逐帧处理
print("Processing video...")
for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # 推理
    results = inference_topdown(model, frame)

    # filter results
    keypoint_score_threshold = 0.3
    min_valid_kpts = 3  # 至少几个关键点得分高于阈值
    min_score = 0.2  # 最小得分阈值
    valid_results = []
    for res in results:
        kpts = res.pred_instances.keypoints  # shape: (num_kpts, 3)
        scores = res.pred_instances.keypoint_scores  # shape: (num_kpts,)
        max_score = scores.max()
        mean_score = scores.mean()
        if mean_score < min_score:
            continue

        valid_kpt_count = (scores > keypoint_score_threshold).sum()
        if valid_kpt_count >= min_valid_kpts:
            valid_results.append(res)

    # 可视化
    for result in valid_results:
        # kpts = res.pred_instances.keypoints  # shape: (num_kpts, 3)
        # track_id = int(res.pred_instances.id[0]) if hasattr(res.pred_instances, 'id') else -1
        # print(res.pred_instances)
        # 过滤掉没有 track_id 的结果
        # if track_id == -1:
        #     continue
        
        # # 获取人的中心位置（nose点或 bbox 中心）
        # if kpts is not None and len(kpts) > 0:
        #     nose_x, nose_y = int(kpts[0][0]), int(kpts[0][1])  # keypoint 0 通常是 nose
        # else:
        #     # 备用方案：用 bbox 中心
        #     bbox = res.pred_instances.bboxes[0]
        #     nose_x = int((bbox[0] + bbox[2]) / 2)
        #     nose_y = int((bbox[1] + bbox[3]) / 2)

        # # 在图像上写上 track_id
        # cv2.putText(
        #     frame,
        #     f'ID: {track_id}',
        #     (nose_x, nose_y - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA
        # )

        visualizer.add_datasample(
            'result',
            frame,
            result,
            draw_bbox=True,
            show_kpt_idx=True,
            show=False,
            wait_time=0,
            out_file=None  # 不保存到文件
        )
    
    # 获取可视化结果图像
    h,w,_ = frame.shape
    vis_frame = visualizer.get_image()[:,w:,:] if valid_results else frame

    # 写入输出视频
    out_video.write(vis_frame)
    # 可选：显示当前帧（调试用）
    cv2.imshow('Pose Estimation', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out_video.release()
print(f"Saved output video to: {output_path}")
