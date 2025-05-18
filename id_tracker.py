import cv2
import torch

from fastreid.engine import DefaultPredictor
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger

class FeatureExtractor(object):
    def __init__(self):
        # build the config
        cfg = get_cfg()
        cfg.merge_from_file("/home/supercomputing/studys/fast-reid/configs/Market1501/AGW_R50.yml")
        cfg.merge_from_list(["MODEL.WEIGHTS", "/home/supercomputing/studys/fast-reid/models/market_agw_R50.pth",
                             "MODEL.DEVICE", "cuda"])
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    def extract(self, original_image):
        # Use the predictor to extract features
        # the model expects RGB inputs
        original_image = original_image[:, :, ::-1]
        # Apply pre-processing to image.
        image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # Make shape with a new batch dimension which is adapted for
        # network input
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        features = self.predictor(image)
        return features
    
if __name__ == "__main__":
    test_image = cv2.imread("/home/supercomputing/studys/fast-reid/datasets/bounding_box_train/0002_c1s1_000451_03.jpg")
    predictor = FeatureExtractor()
    print(predictor.extract(test_image))