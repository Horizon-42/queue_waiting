from id_tracker import IDTracker
from fastreid.evaluation.rerank import re_ranking
import torch
import os
import cv2

def match_across_cameras(tracker1, tracker2):
    objects1 = tracker1.tracked_objects
    objects2 = tracker2.tracked_objects
    matched_pairs = []

    g_feats = torch.cat([obj.features for obj in objects1.values()], dim=0)
    g_ids = [obj.id for obj in objects1.values()]
    g_feats = torch.cat(g_feats, dim=0)
    q_feats = torch.cat([obj.features for obj in objects2.values()], dim=0)
    q_ids = [obj.id for obj in objects2.values()]

    q_g_dist = 1 - torch.matmul(q_feats, g_feats.T)
    q_q_dist = 1 - torch.matmul(q_feats, q_feats.T)
    g_g_dist = 1 - torch.matmul(g_feats, g_feats.T)
    reranked_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    DIST_THRESHOLD = 0.2  # 可调参数：距离小于该阈值才认为是匹配
    for i in range(reranked_dist.shape[0]):
        dist_row = reranked_dist[i]  # 当前特征与所有历史目标的距离
        min_idx = torch.argmin(dist_row)
        min_dist = dist_row[min_idx]

        if min_dist <= DIST_THRESHOLD:
            obj_id = g_ids[min_idx]
            matched_pairs.append((q_ids[i], obj_id))
    return matched_pairs

def run_tracker(tracker, imgs_dir:str, range:tuple[int, int], video_writer=None):
    img_paths = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) if f.endswith('.jpg')]
    # sort the image paths
    img_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    img_paths = img_paths[range[0]:range[1]]
    for img_path in img_paths:
        img = cv2.imread(img_path)
        tracker.track(img)
        img_with_boxes = tracker.draw_boxes(img)
        cv2.imshow("ID Tracker", img_with_boxes)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def pipeline():
    video_res_path = "dataset/res.mp4"
    video_writer = cv2.VideoWriter(video_res_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    tracker1 = IDTracker(camera_id=0)
    run_tracker(tracker1, "dataset/images/start", (0, 300))
    tracker2 = IDTracker(camera_id=1)
    run_tracker(tracker2, "dataset/images/mid", (0, 600))



    tracker3 = IDTracker(camera_id=2)
    run_tracker(tracker1, "dataset/images/end", (0, 1000))

    



if __name__ == "__main__":
    # test_video("dataset/start.mp4")
    # test_images("dataset/images", (0, 10))
    import unittest
    from multiprocessing import Process
    from process_safe_queue import ProcessSafeOrderedDict

    global _shared_dict
    _shared_dict = ProcessSafeOrderedDict()

    class TestProcessSafeOrderedDict(unittest.TestCase):
        def __init__(self, methodName="runTest"):
            super().__init__(methodName)
            self._shared_dict = _shared_dict