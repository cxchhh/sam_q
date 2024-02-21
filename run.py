from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import gradio

sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)


def add_anns(anns, origin_rgb):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    bg = origin_rgb.astype(np.float32)/255
    global_mask = np.zeros((bg.shape[0],bg.shape[1], 3),dtype=np.float32)
    alpha = np.zeros((bg.shape[0],bg.shape[1]))
    for ann in sorted_anns:
        m = ann['segmentation']
        mask_img = np.ones((m.shape[0], m.shape[1], 3),dtype=np.float32)
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            mask_img[:,:,i] = color_mask[i] * m * (1-alpha)
        global_mask = cv2.addWeighted(global_mask, 1, mask_img,1,0)
        alpha = np.logical_or(alpha, m)

    for i in range(3):
        bg[:,:,i] *= 0.65 * alpha + 1.0 * (1-alpha)
    result = cv2.addWeighted(bg,1.0, global_mask,0.35,0)
    
    return np.clip(result,0,1)

def genmask(image):
    masks = mask_generator.generate(image)
    res_img = add_anns(masks, image)
    return res_img
    

demo = gradio.Interface(fn=genmask, inputs="image", outputs="image")
demo.launch()