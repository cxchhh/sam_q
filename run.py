
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