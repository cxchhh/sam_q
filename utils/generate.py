import time
import numpy as np
from utils.utils import add_anns, add_masks
from torchao.quantization import apply_dynamic_quant
from torch._inductor import config as inductorconfig
import torch
sam_checkpoint_b = "./checkpoints/sam_vit_b_01ec64.pth"
sam_checkpoint_l = "./checkpoints/sam_vit_l_0b3195.pth"
sam_checkpoint_h = "./checkpoints/sam_vit_h_4b8939.pth"
current_model_type = "vit_h"

device = "cuda"

def get_model(model_type, chkp, quantize=False):
    if quantize:
        from segment_anything_fast import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    else:
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    model = sam_model_registry[model_type](checkpoint=chkp).cuda()
    model_mask_generator = SamAutomaticMaskGenerator(model)
    model_predictor = SamPredictor(model)   
    if quantize:
        from segment_anything_fast import tools
        tools.apply_eval_dtype_predictor(model_predictor, 'float16')
        for block in model_predictor.model.image_encoder.blocks:
            block.attn.use_rel_pos = True

        apply_dynamic_quant(model_predictor.model.image_encoder)
        apply_dynamic_quant(model_mask_generator.predictor.model)
        inductorconfig.force_fuse_int_mm_with_mul = True
        with torch.no_grad():
            with torch.autograd.profiler.record_function("compilation and warmup"):
                model_predictor.model.image_encoder = torch.compile(
                    model_predictor.model.image_encoder, mode="max-autotune", fullgraph=False)
                # model_mask_generator.predictor.model = torch.compile(
                #     model_mask_generator.predictor.model, mode="max-autotune", fullgraph=False)

    return model, model_predictor, model_mask_generator

sam_q, predictor_q, mask_generator_q = get_model(current_model_type, sam_checkpoint_h, True)
sam, predictor, mask_generator = get_model(current_model_type, sam_checkpoint_h, False)


def change_sam_ver(model_type):
    global sam, sam_q ,mask_generator, predictor, predictor_q, mask_generator_q
    del sam, sam_q

    if model_type == "vit_h" or not model_type:
        sam_q, predictor_q, mask_generator_q = get_model(model_type, sam_checkpoint_h, True)
        sam, predictor, mask_generator = get_model(model_type, sam_checkpoint_h, False)
    elif model_type == "vit_l":
        sam_q, predictor_q, mask_generator_q = get_model(model_type, sam_checkpoint_l, True)
        sam, predictor, mask_generator = get_model(model_type, sam_checkpoint_l, False)
    elif model_type == "vit_b":
        sam_q, predictor_q, mask_generator_q = get_model(model_type, sam_checkpoint_b, True)
        sam, predictor, mask_generator = get_model(model_type, sam_checkpoint_b, False)

    return model_type


def genmask_all(mask_gen,image):
    t0 = time.time()
    masks = mask_gen.generate(image)
    t1 = time.time()
    time_elps = (t1 - t0)
    res_img = add_anns(masks, image)
    return res_img, time_elps

def genmask_points(pred,image,points,mulmask):
    input_points = np.array(points)*np.array([image.shape[1],image.shape[0]])
    input_labels = np.array([1]*len(points))
    pred.set_image(image)
    t0 = time.time()
    masks, scores, logits = pred.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=mulmask,
    )
    t1 = time.time()
    time_elps = (t1 - t0)
    res_img = add_masks(masks, image)
    return res_img, time_elps

def genmask_box(pred,image,box,mulmask):
    sorted_box=[min(box[0],box[2]),min(box[1],box[3]),max(box[0],box[2]),max(box[1],box[3])]
    input_box = np.array(sorted_box)*np.array([image.shape[1],image.shape[0]]*2)
    pred.set_image(image)
    t0 = time.time()
    masks, scores, logits = pred.predict(
        box = input_box,
        multimask_output=mulmask,
    )
    t1 = time.time()
    time_elps = (t1 - t0)
    res_img = add_masks(masks, image)
    return res_img, time_elps

def infer(mtype, image, mode, params, mulmask):
    out = None
    time_elps = 0
    model_mask_gen, model_pred = (mask_generator, predictor) if mtype == 'base' else (mask_generator_q, predictor_q)
    if(mode=='everything'):
        out, time_elps = genmask_all(model_mask_gen,image)
    elif(mode=='points'):
        if len(params) == 0:
            out = image
        else :
            out, time_elps = genmask_points(model_pred, image, params, mulmask)
    elif(mode=='box'):
        if len(params) == 0:
            out = image
        else:
            out, time_elps = genmask_box(model_pred, image, params, mulmask)
    return [out, time_elps]