# import requests
# from PIL import Image
# from optimum.onnxruntime import ORTModelForImageClassification
# from transformers import AutoFeatureExtractor

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# preprocessor = AutoFeatureExtractor.from_pretrained("apple/mobilevit-small")
# model = ORTModelForImageClassification.from_pretrained("apple/mobilevit-small", from_transformers=True)
# inputs = preprocessor(images=image, return_tensors="np")

# outputs = model(**inputs)
# logits = outputs.logits
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])

from transformers import AutoImageProcessor, MobileViTForImageClassification
import torch
from datasets import load_dataset

# from typing import Dict, Optional, Set, Tuple, Union
# from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
# def mobilevit_imageclassification_forward(
#     model,
#     pixel_values: Optional[torch.Tensor] = None,
#     output_hidden_states: Optional[bool] = None,
#     labels: Optional[torch.Tensor] = None,
#     return_dict: Optional[bool] = None,
# ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
#     r"""
#     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#         Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#         config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
#         `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#     """
#     return_dict = return_dict if return_dict is not None else model.config.use_return_dict

#     outputs = model.mobilevit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

#     pooled_output = outputs.pooler_output if return_dict else outputs[1]

#     logits = model.classifier(model.dropout(pooled_output))

#     loss = None
#     if labels is not None:
#         if model.config.problem_type is None:
#             if model.num_labels == 1:
#                 model.config.problem_type = "regression"
#             elif model.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                 model.config.problem_type = "single_label_classification"
#             else:
#                 model.config.problem_type = "multi_label_classification"

#         if model.config.problem_type == "regression":
#             loss_fct = MSELoss()
#             if model.num_labels == 1:
#                 loss = loss_fct(logits.squeeze(), labels.squeeze())
#             else:
#                 loss = loss_fct(logits, labels)
#         elif model.config.problem_type == "single_label_classification":
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, model.num_labels), labels.view(-1))
#         elif model.config.problem_type == "multi_label_classification":
#             loss_fct = BCEWithLogitsLoss()
#             loss = loss_fct(logits, labels)

#     if not return_dict:
#         output = (logits,) + outputs[2:]
#         return ((loss,) + output) if loss is not None else output

#     return ImageClassifierOutputWithNoAttention(
#         loss=loss,
#         logits=logits,
#         hidden_states=outputs.hidden_states,
#     )

import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
print(model)
# for i, m in enumerate(model.modules()):
# 	if type(m) in [torch.nn.Conv2d, torch.nn.Linear]: 
# 		print(i, m)

inputs = image_processor(image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
    # logits = mobilevit_imageclassification_forward(model, **inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])