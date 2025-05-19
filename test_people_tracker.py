from people_extractor import PeopleExtractor
import cv2

def test_box_extractor():
    video_path = "dataset/start.mp4"
    cap = cv2.VideoCapture(video_path)
    people_extractor = PeopleExtractor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 目标检测（只检测人类）
        boxes,poses = people_extractor.extract_pose_and_boxes(frame)

        # 可视化
        for *box, conf, cls in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = people_extractor.draw_pose(frame, poses)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(100) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_box_extractor()