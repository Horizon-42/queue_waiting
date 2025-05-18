import cv2
import mediapipe as mp
from people_extractor import PeopleExtractor

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

video_path = "dataset/start.mp4"
cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)

people_extractor = PeopleExtractor()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes, imgs = people_extractor.extract(image)
    for box, person_image in zip(bboxes, imgs):
        x1, y1, x2, y2 = box
        # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # cv2.putText(image, f'Person', (int(x1), int(y1) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        results = pose.process(person_image)
        person_image = cv2.cvtColor(person_image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(person_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image[int(y1):int(y2), int(x1):int(x2)] = person_image

    cv2.imshow('Pose Estimation', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
