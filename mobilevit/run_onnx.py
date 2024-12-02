import onnx
model_path = "model.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print(f"Before shape inference, the shape info of Y is:\n{onnx_model.graph.value_info}")

from onnx import helper, shape_inference
from onnx import TensorProto
from onnx.tools import update_model_dims

onnx_model = update_model_dims.update_inputs_outputs_dims(onnx_model, {"pixel_values": [1, 3, 224, 224]}, {"logits": [1, 1000]})
inferred_model = shape_inference.infer_shapes(onnx_model, data_prop=True)
print(f"Before shape inference, the shape info of Y is:\n{inferred_model.graph.value_info}")
onnx.save(inferred_model, "inferred_model.onnx")