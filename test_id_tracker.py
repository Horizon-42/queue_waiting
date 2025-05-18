from id_tracker import IDTracker
import cv2

def tet_video(video_path, camera_id:int=0):
    cap = cv2.VideoCapture(video_path)
    id_tracker = IDTracker(camera_id)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # track
        id_tracker.track(frame)

        # draw boxes
        frame_with_boxes = id_tracker.draw_boxes(frame)

        # imshow
        cv2.imshow("ID Tracker", frame_with_boxes)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def test_images(img_dir:str, range:tuple[int, int], camera_id:int=0):
    import os
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    # sort the image paths
    img_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    img_paths = img_paths[range[0]:range[1]]
    id_tracker = IDTracker(camera_id)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        id_tracker.track(img)
        img_with_boxes = id_tracker.draw_boxes(img)
        cv2.imshow("ID Tracker", img_with_boxes)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test video
    # video_path = "dataset/start.mp4"
    # tet_video(video_path)
    # test images
    img_dir = "dataset/images/end"
    test_images(img_dir, (0, 1000),1)
    # test_images(img_dir, (100, 200))