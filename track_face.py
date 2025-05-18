import cv2
import mediapipe as mp

# 初始化 MediaPipe 人脸检测模块
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture("dataset/start.mp4")

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("无法读取摄像头画面")
            break

        # 转换 BGR 图像为 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 检测人脸
        results = face_detection.process(image)

        # 将图像设置为可写并转换回 BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 画出 bounding boxes
        if results.detections:
            for detection in results.detections:
                # 获取边界框
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # 画出边界框
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 可选：显示置信度
                score = int(detection.score[0] * 100)
                cv2.putText(image, f'{score}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow('Face Detection', image)

        # 按 q 键退出
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
