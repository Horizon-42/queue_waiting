from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.visualization import PoseLocalVisualizer
import cv2
from matplotlib import pyplot as plt

register_all_modules()

# config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
# checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
# 使用 bottom-up 模型
# config_file = 'configs/body_2d_keypoint/rtmpose/coco/bottomup-higherhrnet-w32_8xb8-210e_coco-512x512.py'
# checkpoint_file = 'https://download.openmmlab.com/mmpose/bottom_up/higherhrnet/higherhrnet_w32_coco_512x512-cc8c04ad_20210226.pth'


config_file = 'rtmo-l_16xb16-700e_crowdpose-640x640.py' 
checkpoint_file = 'rtmo-l_16xb16-700e_crowdpose-640x640-1008211f_20231211.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'



# please prepare an image with person
results = inference_topdown(model, 'demo.png')

# Load image
image_bgr = cv2.imread('demo.png')

# Initialize visualizer
visualizer = PoseLocalVisualizer()
visualizer.set_dataset_meta(model.dataset_meta)

# Draw pose(s)
for result in results:
    visualizer.add_datasample(
        'result',
        image_bgr,
        result,
        draw_bbox=True,
        show_kpt_idx=False,
        show=False,  # Don't open a GUI window
        wait_time=0,
        out_file='result_vis.jpg'  # Save the result image
    )

# Show with matplotlib
image_rgb = cv2.cvtColor(cv2.imread('result_vis.jpg'), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Pose Estimation Result')
plt.show()