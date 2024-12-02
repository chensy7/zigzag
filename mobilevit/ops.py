from transformers import AutoImageProcessor, MobileViTForImageClassification
import torch
from datasets import load_dataset
from ptflops import get_model_complexity_info
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

inputs = image_processor(image, return_tensors="pt")
logits = model(**inputs).logits
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
print(get_model_complexity_info(model, (3, 256, 256), as_strings=True))