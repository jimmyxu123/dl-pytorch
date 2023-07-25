import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import math

def spatial_pyramid_pool(input,n_batch,input_img_size,patch_number):
    '''
    input: the input tensor [n_batch,channels,height,width]
    n_batch: the number of images in one batch
    input_img_size: the input vector[height,weight]
    patch_number: the list of segmented patches for max pooling layer(eg: [4,2,1] refers to (4,4),(2,2),(1,1)) patches)
    '''
    for i in range(len(patch_number)):
        height = int(math.ceil(input_img_size[0] / patch_number[i]))
        width = int(math.ceil(input_img_size[1] / patch_number[i]))
        h_pad = int((height * patch_number[i] -input_img_size[0] + 1)/2)
        w_pad = int((width * patch_number[i] -input_img_size[1] + 1)/2)
        maxpool = nn.MaxPool2d((height,width),(height,width),(h_pad, w_pad))
        processed_image = maxpool(input)
        flatten_image = processed_image.view(n_batch,-1)
        if i != 0:
            spp = torch.cat((spp,flatten_image),-1)
        else:
            spp = flatten_image
    return spp

class SppNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        n_batch = x.size(0)
        input_img_size = (x.size(-2),x.size(-1))
        patch_number = [4,2,1]
        sppnet = spatial_pyramid_pool(x,n_batch,input_img_size,patch_number)
        return sppnet

