from ultralytics import YOLO
import cv2
import numpy as np

class PeopleExtractor(object):
    def __init__(self):
        self.model = YOLO("models/yolo11m-seg.pt")  # Load a pretrained YOLOv8 model
        self.model.fuse()  # Fuse model layers for faster inference
        self.model.to("cuda")  # Move the model to GPU for faster inference


    def extract(self, original_image):
        results = self.model(original_image, conf=0.5, verbose=False)  # Perform inference on the image
        masks = results[0].masks.data.cpu().numpy()  # Get segmentation masks
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        classes = results[0].boxes.cls.cpu().numpy()
        # get confidence
        confs = results[0].boxes.conf.cpu().numpy()
        # Filter boxes for people (class 0)
        boxes = boxes[classes == 0]
        confs = confs[classes == 0]
        masks = masks[classes == 0]
        # compute human images
        for mask in masks:
            mask = np.array(mask*255, dtype=np.uint8)
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
            mask = np.expand_dims(mask, axis=-1)
            mask = np.concatenate([mask, mask, mask], axis=-1)
            original_image = cv2.bitwise_and(original_image, mask)
            cv2.imshow("mask", mask)
        imgs = []
        for box in boxes:
            x1, y1, x2, y2 = box
            person_image = original_image[int(y1):int(y2), int(x1):int(x2)]
            #TODO 将非mask区域设为0
            imgs.append(person_image)
            cv2.imshow("person", person_image)
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