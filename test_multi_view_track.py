from threading import Thread, Event
from time import sleep
import cv2
from id_tracker_with_deepsort import IDTracker
from queue import Queue, Empty, Full
from dataclasses import dataclass

frame_queue = Queue(maxsize=100)

@dataclass
class FrameInfo:
    frame: any
    camera_id: int

stop_event = Event()

def process_video(tracker:IDTracker, video_path:str):
    global stop_event
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 设置输出视频
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Video {video_path} ended.")
            break

        tracked_frame, tracks = tracker.process_frame(frame)

        frame_info = FrameInfo(frame=tracked_frame, camera_id=tracker.camera_id)
        try:
            # 使用 timeout，如果队列满则阻塞最多 0.1 秒
            # 如果在超时前stop_event被设置，put()不会抛出异常，但会继续阻塞
            # 所以，在put之前和之后都要检查stop_event
            if stop_event.is_set(): # 再次检查，防止在put之前发生中断
                break
            frame_queue.put(frame_info, block=True, timeout=0.1)
        except Full: # 如果队列满且超时，则跳过当前帧
            # print(f"Camera {tracker.camera_id}: Frame queue full, skipping frame {i}")
            pass # 也可以选择不放入队列，直接丢弃帧

    cap.release()

if __name__ == "__main__":
    tracker1 = IDTracker(camera_id=1)
    tracker2 = IDTracker(camera_id=2)
    tracker3 = IDTracker(camera_id=3)

    # 启动两个线程来处理视频
    video_path1 = "dataset/start.mp4"
    video_path2 = "dataset/mid.mp4"
    video_path3 = "dataset/end.mp4"

    thread1 = Thread(target=process_video, args=(tracker1, video_path1))
    thread2 = Thread(target=process_video, args=(tracker2, video_path2))
    thread3 = Thread(target=process_video, args=(tracker3, video_path3))

    all_threads = [thread1, thread2, thread3]
    for thread in all_threads:
        thread.start()

    
    while True:
        try:
            frame_info = frame_queue.get(block=False)
            frame = frame_info.frame
            camera_id = frame_info.camera_id

            cv2.imshow(f"Camera {camera_id}", frame)
            frame_queue.task_done()
        except Empty: # queue is empty
            # 
            all_producers_finished = all(not t.is_alive() for t in all_threads)

            if all_producers_finished:
                # Queue is empty and all producer threads are finished
                print("Main: All producer threads finished and queue is empty. Exiting display loop.")
                break
            else:
                # Queue is empty but producers are still running
                sleep(0.01) 
                continue 
        # 检查 ESC 键
        if cv2.waitKey(1) & 0xFF == 27: # cv2.waitKey返回的是键的ASCII码
            print("Main: ESC pressed, setting stop event.")
            stop_event.set() # 设置停止事件
            break
    
    # Wait for all threads to finish
    for thread in all_threads:
        thread.join()
    cv2.destroyAllWindows()