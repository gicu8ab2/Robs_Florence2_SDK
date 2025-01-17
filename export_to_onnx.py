#https://huggingface.co/amaye15/Florence-2-DaViT-large-ft/resolve/main/export_to_onnx.py


import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
import os
import onnxruntime as ort
import numpy as np

# Constants
MODEL_NAME = "amaye15/DaViT-Florence-2-large-ft"
CACHE_DIR = os.getcwd()
PROMPT = "<OD>"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
ONNX_MODEL_PATH = "model.onnx"

# Load the model and processor
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)

# Prepare the input
image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
inputs = processor(text=PROMPT, images=image, return_tensors="pt")

# Export the model to ONNX
input_names = ["pixel_values"]
output_names = ["output"]
torch.onnx.export(
    model,
    inputs["pixel_values"],
    ONNX_MODEL_PATH,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={"pixel_values": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

# Load the ONNX model
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Prepare the inputs for ONNX model
ort_inputs = {"pixel_values": inputs["pixel_values"].numpy()}

# Run the ONNX model
ort_outs = ort_session.run(None, ort_inputs)

# Display the output
print(ort_outs)
