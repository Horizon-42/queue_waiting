from mmpose.apis import inference_topdown, init_model,inference_bottomup
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer
import cv2

class PoseDetector(object):
    def __init__(self, config_file="rtmo-l_16xb16-700e_crowdpose-640x640.py",
                 checkpoint_file="rtmo-l_16xb16-700e_crowdpose-640x640-1008211f_20231211.pth"):
        # Initialize MMPose modules
        register_all_modules()

        # Load model
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        self.visualizer = PoseLocalVisualizer()
        self.visualizer.set_dataset_meta(self.model.dataset_meta)
        self.keypoint_score_threshold = 0.3
    
    def detect(self, image):
        # Perform inference
        results = inference_bottomup(self.model, image)

        # get the results with max score
        scores_list = []
        for result in results:
            scores = result.pred_instances.keypoint_scores
            scores_list.append(scores.mean())
        max_idx = scores_list.index(max(scores_list))
        return results[max_idx]


    def visualize(self, image, result):
        # print(result)
        # Draw pose(s)

        # draw all keypoints
        kpts = result.pred_instances.keypoints[0]  # shape: (num_kpts, 3)
        
        i = 0
        for kpt in kpts:
            cv2.circle(image, (int(kpt[0]), int(kpt[1])), 3, (0, 255, 0), -1)
            cv2.putText(image, f'{i}', (int(kpt[0]), int(kpt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            i+=1

        return image

    

if __name__ == "__main__":
    from people_extractor import PeopleExtractor
    # Initialize PoseDetector
    pose_detector = PoseDetector()
    people_extractor = PeopleExtractor()

    # video_path = 'dataset/end.mp4'
    video_path = 'dataset/start.mp4'

    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # iterate through the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # detect people
        bboxes, imgs = people_extractor.extract(frame)
        for box, person_image in zip(bboxes, imgs):
            x1, y1, x2, y2 = box
            # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.putText(image, f'Person', (int(x1), int(y1) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            result = pose_detector.detect(person_image)
            person_image = pose_detector.visualize(person_image, result)
            frame[int(y1):int(y2), int(x1):int(x2)] = person_image
        # Show the image
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()