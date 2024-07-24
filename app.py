import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Taiwan Common Wild Poisonous Plant Classifier"
description = "Dataset of Common Toxic Plants in the Wild in Taiwan from Bing Image Search. The Accuracy of This Model is About 80%."
article="<p style='text-align: center'><a href='https://github.com/ILoveSorasakiHina/Taiwan-Common-Wild-Poisonous-Plant-Classifier' target='_blank'>Blog post</a></p>"
examples = ['巴豆.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()