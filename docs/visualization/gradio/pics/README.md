```python
import gradio as gr
from transformers import pipeline
from ultralytics import YOLO
from skimage import data
from PIL import Image
import pandas as pd

model = YOLO('yolov8n-cls.pt')

# 用三张图片作为一个实例
Image.fromarray(data.coffee()).save('coffee.jpeg')
Image.fromarray(data.astronaut()).save('people.jpeg')
Image.fromarray(data.cat()).save('cat.jpeg')

def predict(img):
    result = model.predict(source=img)
    df = pd.Series(result[0].names).to_frame()
    df = pd.Series(result[0].names).to_frame()
    df.columns = ['names']
    df['probs'] = result[0].probs.cpu().numpy().data
    df = df.sort_values('probs',ascending=False)
    res = dict(zip(df['names'],df['probs']))
    return res

gr.close_all()

# 输出5个最大可能类别
demo = gr.Interface(fn=predict, inputs=gr.Image(type='pil'), outputs=gr.Label(num_top_classes=5),
                    examples = ['coffee.jpeg', 'people.jpeg', 'cat.jpeg'])
demo.launch(share=True)
```