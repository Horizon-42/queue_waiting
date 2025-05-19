from ultralytics import YOLO
import cv2
import numpy as np

class PeopleExtractor(object):
    def __init__(self):
        self.model = YOLO("models/yolo11n-pose.pt")  # Load a pretrained YOLOv8 model
        self.model.fuse()  # Fuse model layers for faster inference
        self.model.to("cuda")  # Move the model to GPU for faster inference


    def extract(self, original_image):
        results = self.model(original_image, conf=0.5, verbose=False)  # Perform inference on the image
        # masks = results[0].masks.data.cpu().numpy()  # Get segmentation masks
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        
        classes = results[0].boxes.cls.cpu().numpy()
        # get confidence
        confs = results[0].boxes.conf.cpu().numpy()
        mask = (confs > 0.7) & (classes == 0)
        # Filter boxes for people (class 0)
        boxes = boxes[mask]
        confs = confs[mask]
        # masks = masks[classes == 0]
        imgs = []
        for box in boxes:
            x1, y1, x2, y2 = box
            person_image = original_image[int(y1):int(y2), int(x1):int(x2)]
            #TODO 将非mask区域设为0
            imgs.append(person_image)
        return boxes, imgs
    
    def extract_boxes(self, original_image):
        results = self.model(original_image, conf=0.5, verbose=False)  # Perform inference on the image
        # masks = results[0].masks.data.cpu().numpy()  # Get segmentation masks
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        
        classes = results[0].boxes.cls.cpu().numpy()
        # get confidence
        confs = results[0].boxes.conf.cpu().numpy()
        mask = (confs > 0.7) & (classes == 0)
        # Filter boxes for people (class 0)
        boxes = boxes[mask]
        confs = confs[mask]
        classes = classes[mask]

        # horizontal stack boxes, confs, classes
        boxes = np.hstack((boxes, confs.reshape(-1, 1), classes.reshape(-1, 1)))
        return boxes
    
    def extract_pose_and_boxes(self, original_image):
        results = self.model(original_image, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Get bounding boxes
        classes = results[0].boxes.cls.cpu().numpy()
        # get confidence
        confs = results[0].boxes.conf.cpu().numpy()
        mask = (confs > 0.7) & (classes == 0)
        # Filter boxes for people (class 0)
        boxes = boxes[mask]
        confs = confs[mask]
        classes = classes[mask]

        # get pose
        poses = results[0].keypoints.cpu().numpy()
        # Filter poses for people (class 0)
        poses = poses[mask].data
        print(poses)
        # horizontal stack boxes, confs, classes
        boxes = np.hstack((boxes, confs.reshape(-1, 1), classes.reshape(-1, 1)))
        return boxes, poses


    def draw_boxes(self, image, boxes):

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return image

    def draw_pose(self, image, poses):
        for pose in poses:
            i = 0
            for point in pose:
                x, y, conf = point
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(image, f"id{i}: {conf:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                i+= 1
        return image
    
