from id_tracker_with_deepsort import IDTracker
from thread_safe_ordered_dict import ThreadSafeOrderedDict, TrackInfo, Person
from queue import Queue, Empty, Full
from dataclasses import dataclass
from threading import Thread, Event
import cv2
import time
import numpy as np
from typing import List, Tuple

@dataclass
class FrameInfo:
    frame: any
    camera_id: int

class MultiViewTracker:

    def __init__(self):
        self.trackers: List[IDTracker] = []
        self.frame_queue = Queue(maxsize=100)
        self.stop_event = Event()
        self.global_tracks = ThreadSafeOrderedDict()
    
    def __add_tracker(self, camera_id: int):
        tracker = IDTracker(camera_id=camera_id, global_tracks=self.global_tracks)
        self.trackers.append(tracker)
        return tracker
    
    def __process_view(self, tracker:IDTracker, video_stream:str):
        cap = cv2.VideoCapture(video_stream)
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Video {video_stream} ended.")
                break

            tracked_frame, tracks = tracker.process_frame(frame)

            frame_info = FrameInfo(frame=tracked_frame, camera_id=tracker.camera_id)
            try:
                if self.stop_event.is_set():
                    break
                self.frame_queue.put(frame_info, block=True, timeout=0.1)
            except Full:
                # drop the frame if queue is full
                pass
        cap.release()
    
    def __display_frames(self):
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame_info = self.frame_queue.get(block=True, timeout=0.1)
                frame = frame_info.frame
                camera_id = frame_info.camera_id

                cv2.imshow(f"Camera {camera_id}", frame)
                self.frame_queue.task_done()
            except Empty:
                time.sleep(0.01)  # queue is empty, wait a bit
                continue
            
            # check ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                print("Main: ESC pressed, setting stop event.")
                self.stop_event.set()
                break
        cv2.destroyAllWindows()

    def start_tracking(self, video_streams: List[str]):
        threads = []
        for i, video_stream in enumerate(video_streams):
            tracker = self.__add_tracker(camera_id=i)
            thread = Thread(target=self.__process_view, args=(tracker, video_stream))
            threads.append(thread)
            thread.start()

        display_thread = Thread(target=self.__display_frames)
        display_thread.start()

        for thread in threads:
            thread.join()
        
        # stop if ESC is pressed or all threads are done
        print("Main: Stopping all threads.")
        self.stop_event.set()
        display_thread.join()
