
import gradio
from utils.utils import jsfile
from utils.generate import infer

theme=gradio.themes.Default(primary_hue=gradio.themes.colors.blue, secondary_hue=gradio.themes.colors.blue).set(
    button_primary_background_fill="*secondary_200",
    button_primary_background_fill_hover="*secondary_300",
)
blocks = gradio.Blocks(theme = theme,js=jsfile("src/init.js"))
with blocks as demo:
    with gradio.Row(equal_height=True):
        with gradio.Column(scale=1,min_width=120):
            model_type = gradio.Radio(label="backbone",choices=["vit_h","vit_l", "vit_b"], value="vit_h")
            model_type.change(lambda x: x, model_type,[],js="(x)=>{globalThis.model_type=x;}")
        with gradio.Column(scale=5):
            inputImg = gradio.Image(label="input",type="numpy")
            with gradio.Row():
                mode = gradio.Radio(label="prediction type",choices=[ "everything","points", "box"], value="everything",scale=3)
                mode.change(lambda x: x, mode,[],js="(x)=>{globalThis.mode=x; globalThis.resetCanvas();}")
                multimask = gradio.Checkbox(label="multimask",value=False,scale=1)
            submitBtn = gradio.Button("submit")
            inputImg.upload(fn=lambda x: "everything", inputs=inputImg, outputs=mode, js="()=>{globalThis.resetCanvas();}")
        with gradio.Column(scale=5):
            outputImg = gradio.Image(label="output",type="numpy")
    
    # the second 'mode' is a placeholder
    submitBtn.click(fn=infer, inputs=[model_type,inputImg, mode, mode, multimask], outputs=outputImg,js="(mt,i,m,p,mul) => globalThis.submit(mt,i,m,p,mul)")

if __name__ == "__main__":
    demo.launch()