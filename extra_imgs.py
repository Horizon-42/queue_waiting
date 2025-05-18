import cv2
import numpy as np
import os

video_path = "dataset/end.mp4"
image_dir = "dataset/images/end"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

cap = cv2.VideoCapture(video_path)
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save the frame as an image
    image_path = os.path.join(image_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(image_path, frame)
    frame_count += 1
    # print(f"Saved {image_path}")
cap.release()
cv2.destroyAllWindows()