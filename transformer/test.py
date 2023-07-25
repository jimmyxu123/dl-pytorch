import torch
import numpy as np
import vision_transformer 
from vision_transformer import *

model = Vit(
    img_size= 256,
    patch_size = 32,
    n_class = 2,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)
model.eval()
torch.save(model.state_dict(), '/Users/jimmyxu/Desktop/transformer/model.pth')



from torchvision import transforms
from PIL import Image
import sys,os
os.chdir(sys.path[0])
img_pth = '/Users/jimmyxu/Desktop/cat.jpg'
input = Image.open(img_pth)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((256, 256)),
                                            normalize])
input = trans(input)
print(model(input.unsqueeze(0)))