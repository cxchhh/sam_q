import gradio
from utils.utils import jsfile
from utils.generate import infer, change_sam_backbone
theme=gradio.themes.Default(primary_hue=gradio.themes.colors.blue, secondary_hue=gradio.themes.colors.blue).set(
    button_primary_background_fill="*secondary_200",
    button_primary_background_fill_hover="*secondary_300",
)
blocks = gradio.Blocks(theme = theme,js=jsfile("src/init.js"))
with blocks as demo:
    with gradio.Row(equal_height=True):

        with gradio.Column(scale=1,min_width=120):
            model_type = gradio.Radio(label="backbone",choices=["vit_h","vit_l", "vit_b"], value="vit_h")
            model_type.change(change_sam_backbone, model_type,model_type,js="(x)=>{globalThis.model_type=x; return x;}")
        
        with gradio.Column(scale=5):
            inputImg = gradio.Image(label="input",type="numpy")
            
            with gradio.Row():
                mode = gradio.Radio(label="prediction type",choices=[ "everything","points", "box"], value="everything",scale=3)
                mode.change(lambda x: x, mode,[],js="(x)=>{globalThis.mode=x; globalThis.resetCanvas();}")
                multimask = gradio.Checkbox(label="multimask",value=False,scale=1)
            
            submitBtn = gradio.Button("submit")
            inputImg.upload(fn=lambda x: "everything", inputs=inputImg, outputs=mode, js="()=>{globalThis.resetCanvas();}")
        
        with gradio.Column(scale=5):
            with gradio.Row():
                outputImg = gradio.Image(label="base model output",type="numpy")
                results = gradio.Textbox(interactive=False,label="base model inference time:",max_lines=1)
            
            with gradio.Row():
                outputImg_q = gradio.Image(label="quantized model output",type="numpy")
                results_q = gradio.Textbox(interactive=False,label="quantized model inference time:",max_lines=1)
    
    # the second 'mode' is a placeholder
    submitBtn.click(fn=infer, inputs=[model_type,inputImg, mode, mode, multimask], outputs=[outputImg, results],
                    js="(mt,i,m,p,mul) => globalThis.submit('base',i,m,p,mul)",).then(
                    fn=infer, inputs=[model_type,inputImg, mode, mode, multimask], outputs=[outputImg_q, results_q],
                    js="(mt,i,m,p,mul) => globalThis.submit('quant',i,m,p,mul)")
    
if __name__ == "__main__":
    demo.launch()