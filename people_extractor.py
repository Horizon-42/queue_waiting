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
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        # long_boxes = (boxes[:,3] - boxes[:,1])>300  # Calculate the height of the bounding boxes
        # boxes[long_boxes,3] = boxes[long_boxes,1] + 0.5*(boxes[long_boxes,3] - boxes[long_boxes,1])  # Adjust the height of the bounding boxes
        # # get classes
        classes = results[0].boxes.cls.cpu().numpy()
        # get confidence
        confs = results[0].boxes.conf.cpu().numpy()
        # Filter boxes for people (class 0)
        boxes = boxes[classes == 0]
        confs = confs[classes == 0]
        classes = classes[classes == 0]
        # return np.hstack((boxes, confs[:, np.newaxis], classes[:, np.newaxis]))  # Combine boxes, confidence, and classes
        # compute human images
        imgs = []
        for box in boxes:
            x1, y1, x2, y2 = box
            person_image = original_image[int(y1):int(y2), int(x1):int(x2)]
            imgs.append(person_image)
        return boxes, imgs
    def draw_boxes(self, image, boxes):

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
    
if __name__ == "__main__":
    video_path = "dataset/start.mp4"
    cap = cv2.VideoCapture(video_path)
    people_extractor = PeopleExtractor()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = people_extractor.extract(frame)
        frame_with_boxes = people_extractor.draw_boxes(frame, boxes)

        cv2.imshow("People Detection", frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()