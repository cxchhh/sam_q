# Quantization of SAM with gradio demo

![](./assets/img.png)

### usage
Clone this repo to your local directory, and download the ViT weights using following command:
```
./download.sh
```
and then create the python environment:
```
conda create -n samq python=3.10
```
```
conda activate samq
```
```
pip install -r requirements.txt
```
run the model:
```
python run.py
```

It will be quite a slow start for the first time because it needs to set up the optimal quantization parameters. After that, it will start faster.