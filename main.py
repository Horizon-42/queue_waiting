from id_tracker import IDTracker
from fastreid.evaluation.rerank import re_ranking
import torch
import os
import cv2
import numpy as np

def match_across_cameras(tracker1, tracker2):
    objects1 = tracker1.tracked_objects
    objects2 = tracker2.tracked_objects
    matched_pairs = []

    g_feats = torch.cat([obj.features for obj in objects1.values()], dim=0)
    g_ids = [obj.id for obj in objects1.values()]
    q_feats = torch.cat([obj.features for obj in objects2.values()], dim=0)
    q_ids = [obj.id for obj in objects2.values()]

    q_g_dist = 1 - torch.matmul(q_feats, g_feats.T)
    q_q_dist = 1 - torch.matmul(q_feats, q_feats.T)
    g_g_dist = 1 - torch.matmul(g_feats, g_feats.T)
    reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    DIST_THRESHOLD = 0.2  # 可调参数：距离小于该阈值才认为是匹配
    for i in range(reranked_dist.shape[0]):
        dist_row = reranked_dist[i]  # 当前特征与所有历史目标的距离
        min_idx = np.argmin(dist_row)
        min_dist = dist_row[min_idx]

        if min_dist <= DIST_THRESHOLD:
            obj_id = g_ids[min_idx]
            matched_pairs.append((q_ids[i], obj_id))
    return matched_pairs

def get_img_paths(imgs_dir:str, range:tuple[int, int]):
    img_paths = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg')]
    # sort the image paths
    img_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    img_paths = img_paths[range[0]:range[1]]
    return img_paths

def run_tracker(tracker, imgs_dir:str, range:tuple[int, int], video_writer):
    img_paths = get_img_paths(imgs_dir, range)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        tracker.track(img)
        img_with_boxes = tracker.draw_boxes(img)
        video_writer.write(img_with_boxes)
        cv2.imshow("ID Tracker", img_with_boxes)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

# def re_show(tracker, imgs_dir:str, range:tuple[int, int], video_writer):
#     img_paths = get_img_paths(imgs_dir, range)
#     for img_path in img_paths:
#         img = cv2.imread(img_path)
#         tracker.track(img)
#         img_with_boxes = tracker.draw_boxes(img)
#         video_writer.write(img_with_boxes)
#         cv2.imshow("ID Tracker", img_with_boxes)
#         if cv2.waitKey(1) == 27:
#             break
#     cv2.destroyAllWindows()

def pipeline():
    # TODO multi process running
    process_video_path = "dataset/process.mp4"
    process_video_writer = cv2.VideoWriter(process_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

    tracker1 = IDTracker(camera_id=0)
    run_tracker(tracker1, "dataset/images/start", (0, 300), process_video_writer)
    tracker2 = IDTracker(camera_id=1)
    run_tracker(tracker2, "dataset/images/mid", (0, 550), process_video_writer)
    matched_pairs = match_across_cameras(tracker1, tracker2)
    # update the object IDs in tracker2 based on matched pairs
    for q_id, g_id in matched_pairs:
        if g_id in tracker1.tracked_objects:
            tracker2.tracked_objects[q_id].id = g_id

    tracker3 = IDTracker(camera_id=2)
    run_tracker(tracker3, "dataset/images/end", (0, 800),process_video_writer)
    matched_pairs = match_across_cameras(tracker2, tracker3)
    # update the object IDs in tracker3 based on matched pairs
    for q_id, g_id in matched_pairs:
        if g_id in tracker2.tracked_objects:
            tracker3.tracked_objects[q_id].id = g_id

    process_video_writer.release()

    res_video_path = "dataset/res.mp4"
    res_video_writer = cv2.VideoWriter(res_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    run_tracker(tracker1, "dataset/images/start", (0, 600), res_video_writer)
    run_tracker(tracker2, "dataset/images/mid", (0, 550), res_video_writer)
    run_tracker(tracker3, "dataset/images/end", (0, 700), res_video_writer)
    res_video_writer.release()

    # person count
    person_count = len(tracker1.tracked_objects) + len(tracker2.tracked_objects) + len(tracker3.tracked_objects)
    print(f"Total number of unique persons tracked: {person_count}")

    # average time estimation
    averg_frame_count = 0
    for tracker in [tracker1, tracker2, tracker3]:
        for obj in tracker.tracked_objects.values():
            averg_frame_count += (obj.last_seen - obj.first_seen)
    averg_frame_count /= person_count
    print(f"Average time estimation for each person: {averg_frame_count} frames")


if __name__ == "__main__":
    pipeline()