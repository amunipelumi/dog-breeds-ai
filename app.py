import gradio as gr
import os
import torch

from model import create_model
from timeit import default_timer as timer
from typing import Tuple, Dict

with open("class_names.txt", "r") as f:
  class_names = [i.strip() for i in  f.readlines()]

model, auto_transforms, _ = create_model(len(class_names))

model.load_state_dict(
    torch.load(
        f="effnetb3_dog_vision.pth",
        map_location=torch.device("cpu")
    )
) 

def predict(img) -> Tuple[Dict, float]:
  """Transforms and performs a prediction on img and returns prediction and time taken.
  """

  start_time = timer()

  img = auto_transforms(img).unsqueeze(0)

  model.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(model(img), dim=1)

  pred_labels_and_probs = {class_names[i].title().replace("_", " "): float(pred_probs[0][i]) for i in range(len(class_names))}

  pred_time = round(timer() - start_time, 2)

  return pred_labels_and_probs, pred_time


title = "Dog Breed classifier" 
description = f"This CV model aims to classify over 100 classes of various dog breeds. Utilizing EfficientNet as backbone."
article = "© Amuni Pelumi https://www.amunipelumi.com/"

example_list = [["examples/" + example] for example in os.listdir("examples")]

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=1, label="Top (5) Predictions"), 
                             gr.Number(label="Prediction Duration (S)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch() 
