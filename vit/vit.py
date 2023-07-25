import torch
import torch.nn as nn
from vit_pytorch import ViT
import os
from onnxsim import simplify

model = ViT(
    image_size= 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

model.eval()
onnx_path = './Desktop/vit.onnx'
input_tensor = torch.randn(1, 3, 256, 256)
try:
    torch.onnx.export(model.to('cpu'),
                      input_tensor.to('cpu'),
                      onnx_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=False,
                      input_names=['input'],
                      output_names=['output'])
except:
    import traceback
    traceback.print_exc()
import onnx
onnx_model = onnx.load(onnx_path)
model_s, check = simplify(onnx_model)
onnx.save(model_s, onnx_path.replace('vit.onnx','vit_simplified.onnx'))



