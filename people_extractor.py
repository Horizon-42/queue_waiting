from ultralytics import YOLO
import cv2
import numpy as np

class PeopleExtractor(object):
    def __init__(self):
        self.model = YOLO("models/yolo11n.pt")  # Load a pretrained YOLOv8 model
        self.model.fuse()  # Fuse model layers for faster inference
        self.model.to("cuda")  # Move the model to GPU for faster inference


    def extract(self, original_image):
        results = self.model(original_image, conf=0.5, verbose=False)  # Perform inference on the image
        # masks = results[0].masks.data.cpu().numpy()  # Get segmentation masks
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        # 计算每个框的高度
        # heights = boxes[:, 3] - boxes[:, 1]
        # widths = boxes[:, 2] - boxes[:, 0]

        # # # 找到高度大于 300 的框
        # mask = heights > widths*2.3

        # # # 计算新的 y2 值：减少高度的一半，但保持总高度不少于 300
        # new_heights = heights[mask] / 2
        # boxes[mask, 3] = boxes[mask, 1] + new_heights
        
        classes = results[0].boxes.cls.cpu().numpy()
        # get confidence
        confs = results[0].boxes.conf.cpu().numpy()
        # Filter boxes for people (class 0)
        boxes = boxes[classes == 0]
        confs = confs[classes == 0]
        # masks = masks[classes == 0]
        imgs = []
        for box in boxes:
            x1, y1, x2, y2 = box
            person_image = original_image[int(y1):int(y2), int(x1):int(x2)]
            #TODO 将非mask区域设为0
            imgs.append(person_image)
        return boxes, imgs
    def draw_boxes(self, image, boxes):

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return image
    
if __name__ == "__main__":
    video_path = "dataset/start.mp4"
    cap = cv2.VideoCapture(video_path)
    people_extractor = PeopleExtractor()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, _ = people_extractor.extract(frame)
        frame_with_boxes = people_extractor.draw_boxes(frame, boxes)

        cv2.imshow("People Detection", frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()