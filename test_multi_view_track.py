from multi_view_tracker import MultiViewTracker

if __name__ == "__main__":
    video_path1 = "dataset/start.mp4"
    video_path2 = "dataset/mid.mp4"
    video_path3 = "dataset/end.mp4"

    multi_tracker = MultiViewTracker()

    multi_tracker.start_tracking([video_path1, video_path2, video_path3])