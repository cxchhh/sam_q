from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import cv2
import gradio
from utils.utils import jsfile, add_anns, add_masks

sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

def setEmbedding(image):
    predictor.set_image(image)
    

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


theme=gradio.themes.Default(primary_hue=gradio.themes.colors.gray, secondary_hue=gradio.themes.colors.blue).set(
    button_primary_background_fill="*secondary_200",
    button_primary_background_fill_hover="*secondary_300",
)
blocks = gradio.Blocks(theme = theme,js=jsfile("src/init.js"))
with blocks as demo:
    with gradio.Row(equal_height=True):
        with gradio.Column():
            inputImg = gradio.Image(label="input",type="numpy")
            with gradio.Row():
                mode = gradio.Radio(label="prediction type",choices=[ "everything","points", "box"], value="everything")
                mode.change(lambda x: x, mode,[],js="(x)=>{globalThis.mode=x; globalThis.resetCanvas();}")
                multimask = gradio.Checkbox(label="multimask",value=False)
            submitBtn = gradio.Button("submit")
            inputImg.upload(fn=lambda x: "everything", inputs=inputImg, outputs=mode, js="()=>{globalThis.resetCanvas();}")
        with gradio.Column():
            outputImg = gradio.Image(label="output",type="numpy")
    
    # the second 'mode' is a placeholder
    submitBtn.click(fn=infer, inputs=[inputImg, mode, mode, multimask], outputs=outputImg,js="(i,m,p,mul) => globalThis.submit(i,m,p,mul)")

if __name__ == "__main__":
    demo.launch()