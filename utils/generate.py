from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
from utils.utils import add_anns, add_masks
sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

def genmask_all(image):
    masks = mask_generator.generate(image)
    res_img = add_anns(masks, image)
    return res_img

def genmask_points(image,points,mulmask):
    input_points = np.array(points)*np.array([image.shape[1],image.shape[0]])
    input_labels = np.array([1]*len(points))
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=mulmask,
    )
    res_img = add_masks(masks, image)
    return res_img

def genmask_box(image,box,mulmask):
    sorted_box=[min(box[0],box[2]),min(box[1],box[3]),max(box[0],box[2]),max(box[1],box[3])]
    input_box = np.array(sorted_box)*np.array([image.shape[1],image.shape[0]]*2)
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        box = input_box,
        multimask_output=mulmask,
    )
    res_img = add_masks(masks, image)
    return res_img

def infer(image, mode, params, mulmask):
    if(mode=='everything'):
        return genmask_all(image)
    elif(mode=='points'):
        if len(params) == 0:
            return image
        return genmask_points(image, params, mulmask)
    elif(mode=='box'):
        if len(params) == 0:
            return image
        return genmask_box(image, params, mulmask)