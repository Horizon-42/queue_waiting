import cv2

#@markdown To better demonstrate the Pose Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


if __name__ == "__main__":
    video_path = 'dataset/IMG_1797.MOV'

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    cv2.namedWindow("frame", cv2.WINDOW_GUI_NORMAL)

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap image for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect
        detection_result = detector.detect(mp_image)

        # Draw landmarks
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)

        # Convert RGB back to BGR for OpenCV display
        bgr_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", bgr_annotated)

        if cv2.waitKey(1) == 27:
            break