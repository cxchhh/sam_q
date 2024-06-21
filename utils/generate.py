import time, os

import pandas as pd
os.environ['SEGMENT_ANYTHING_FAST_USE_FLASH_4'] = '0'
os.environ['FAST_USE_FLASH_4'] = '0'
import numpy as np
from utils.utils import add_anns, add_masks
from torchao.quantization import apply_dynamic_quant
from torch._inductor import config as inductorconfig
import torch
torch.set_default_tensor_type(torch.FloatTensor)

from matplotlib import pyplot as plt

chkpt_path = "./checkpoints"

sam_checkpoint_b = f"{chkpt_path}/sam_vit_b_01ec64.pth"
sam_checkpoint_l = f"{chkpt_path}/sam_vit_l_0b3195.pth"
sam_checkpoint_h = f"{chkpt_path}/sam_vit_h_4b8939.pth"
current_model_type = "vit_h"

device = "cuda"


def get_model(model_type, chkp, quantize=False):
    if quantize:
        from segment_anything_fast import sam_model_registry
        from utils.automask_generator import SamAutomaticMaskGenerator
        model = sam_model_registry[model_type](checkpoint=chkp).cuda()
        model_mask_generator = SamAutomaticMaskGenerator(model)
        model_predictor = model_mask_generator.predictor

        from segment_anything_fast import tools
        tools.apply_eval_dtype_predictor(model_predictor, torch.bfloat16)

        for block in model_predictor.model.image_encoder.blocks:
            block.attn.use_rel_pos = True

        apply_dynamic_quant(model_predictor.model.image_encoder)
        inductorconfig.force_fuse_int_mm_with_mul = True
    
        with torch.no_grad():
            with torch.autograd.profiler.record_function("compilation and warmup"):
                model_predictor.model.image_encoder = torch.compile(
                    model_predictor.model.image_encoder, mode="max-autotune-no-cudagraphs", fullgraph=False)
        # warm-up        
        model_predictor.set_image(np.zeros([512,512,3], np.uint8))

    else:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        model = sam_model_registry[model_type](checkpoint=chkp).cuda()
        model_mask_generator = SamAutomaticMaskGenerator(model)
        model_predictor = model_mask_generator.predictor
        
    return model, model_predictor, model_mask_generator

ori_mem = torch.cuda.memory_allocated()
sam_q, predictor_q, mask_generator_q = get_model(current_model_type, sam_checkpoint_h if current_model_type =="vit_h" 
                                                 else sam_checkpoint_l if current_model_type =="vit_l"
                                                 else sam_checkpoint_b
                                                , True)
sam_q_mem = torch.cuda.memory_allocated()
sam, predictor, mask_generator = get_model(current_model_type, sam_checkpoint_h if current_model_type =="vit_h" 
                                                 else sam_checkpoint_l if current_model_type =="vit_l"
                                                 else sam_checkpoint_b, False)
sam_mem = torch.cuda.memory_allocated()
sam_size = sam_mem - sam_q_mem
sam_q_size = sam_q_mem - ori_mem

def change_sam_backbone(model_type):
    global sam, sam_q ,mask_generator, predictor, predictor_q, mask_generator_q
    global ori_mem, sam_q_mem, sam_mem, sam_size, sam_q_size
    if sam:
        del sam
        del sam_q  
    torch.cuda.empty_cache()
    # ori_mem = torch.cuda.memory_allocated()
    if model_type == "vit_h" or not model_type:
        sam_q, predictor_q, mask_generator_q = get_model(model_type, sam_checkpoint_h, True)
        # sam_q_mem = torch.cuda.memory_allocated()
        sam, predictor, mask_generator = get_model(model_type, sam_checkpoint_h, False)
        # sam_mem = torch.cuda.memory_allocated()
    elif model_type == "vit_l":
        sam_q, predictor_q, mask_generator_q = get_model(model_type, sam_checkpoint_l, True)
        # sam_q_mem = torch.cuda.memory_allocated()
        sam, predictor, mask_generator = get_model(model_type, sam_checkpoint_l, False)
        # sam_mem = torch.cuda.memory_allocated()
    elif model_type == "vit_b":
        sam_q, predictor_q, mask_generator_q = get_model(model_type, sam_checkpoint_b, True)
        # sam_q_mem = torch.cuda.memory_allocated()
        sam, predictor, mask_generator = get_model(model_type, sam_checkpoint_b, False)
        # sam_mem = torch.cuda.memory_allocated()

    # sam_size = sam_mem - sam_q_mem
    # sam_q_size = sam_q_mem - ori_mem

    print(f'sam_size:{sam_size} sam_q_size{sam_q_size}')


    print("backbone changed to",model_type)
    return model_type

def genmask_all(mask_gen,image,quantize):
    torch.cuda.empty_cache()
    
    torch.cuda.synchronize();t0 = time.time()
    masks = mask_gen.generate(image)
    torch.cuda.synchronize();t1 = time.time()
    time_elps = (t1 - t0)
    res_img = add_anns(masks, image)
    return res_img, time_elps

def genmask_points(pred,image,points,mulmask,qauntize):
    input_points = np.array(points)*np.array([image.shape[1],image.shape[0]])
    input_labels = np.array([1]*len(points))

    torch.cuda.synchronize();t0 = time.time()
    pred.set_image(image)
    masks, scores, logits = pred.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=mulmask,
    )
    torch.cuda.synchronize();t1 = time.time()
    time_elps = (t1 - t0)
    res_img = add_masks(masks, image)
    return res_img, time_elps

def genmask_box(pred,image,box,mulmask,quantize):
    sorted_box=[min(box[0],box[2]),min(box[1],box[3]),max(box[0],box[2]),max(box[1],box[3])]
    input_box = np.array(sorted_box)*np.array([image.shape[1],image.shape[0]]*2)

    torch.cuda.synchronize();t0 = time.time()
    pred.set_image(image)
    masks, scores, logits = pred.predict(
        box = input_box,
        multimask_output=mulmask,
    )
    torch.cuda.synchronize();t1 = time.time()
    time_elps = (t1 - t0)
    res_img = add_masks(masks, image)
    return res_img, time_elps

base_time = 0
base_mem = 0
quant_time = 0
quant_mem = 0

def infer(mtype, image: np.ndarray, mode, params, mulmask):
    global base_time, base_mem, quant_time, quant_mem
    out = None
    time_elps = 0
    image = image.astype(np.float32)
    model_mask_gen, model_pred = (mask_generator, predictor) if mtype == 'base' else (mask_generator_q, predictor_q)
    if(mode=='everything'):
        out, time_elps = genmask_all(model_mask_gen,image,(mtype=='quant'))
    elif(mode=='points'):
        if len(params) == 0:
            out = image
        else :
            out, time_elps = genmask_points(model_pred, image, params, mulmask,(mtype=='quant'))
    elif(mode=='box'):
        if len(params) == 0:
            out = image
        else:
            out, time_elps = genmask_box(model_pred, image, params, mulmask,(mtype=='quant'))
    time_elps = (time_elps * 1000)
    mem = sam_size if mtype == 'base' else sam_q_size

    if mtype == "base":
        base_time = time_elps
        base_mem = mem/1024/1024/1024
    else:
        quant_time = time_elps
        quant_mem = mem/1024/1024/1024

    return out

def plot_bars():
    global base_time, base_mem, quant_time, quant_mem
    return [pd.DataFrame({
            "models": ['base model',
                   'quantized model'],
            "inference time / ms": [base_time, quant_time]
            }), pd.DataFrame({
            "models": ['base model',
                   'quantized model'],
            "memory size / GB": [base_mem, quant_mem]
            })]