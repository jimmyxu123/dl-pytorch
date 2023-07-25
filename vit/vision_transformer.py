import torch
import transformer
from transformer import *
import torch.nn as nn
import copy
from loguru import logger

class TransformerEncoder(nn.Module):
    def __init__(self,input):
        super().__init__()
        self.input = input
    
    def forward(self, input, heads, mlp_dim, depth):
        dropout = 0.1
        c = copy.deepcopy
        attn = transformer.MultiHeadAttention(heads,input.size(-1))
        ff = transformer.PositionWiseFeedForward(input.size(-1),mlp_dim,dropout)
        encode = transformer.Encoder(transformer.EncoderLayer(input.size(-1),c(attn),c(ff),dropout), depth)
        output = encode(input, mask=None)
        return output

class generator(nn.Module):
    def __init__(self, d_model, n_class):
        super().__init__()
        self.n_class = n_class

    def forward(self,x):
        x = x.mean(dim=1)
        d_model = x.size(-1)
        mlp = nn.Sequential(nn.LayerNorm(d_model),
                            nn.Linear(d_model,self.n_class))
        return F.log_softmax(mlp(x),dim=-1)

class ImagePreprocess(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
    
    def pair(self,x):
        return x if isinstance(x, tuple) else (x,x)
    
    def forward(self,x):
        img_h, img_w = self.pair(self.img_size)
        patch_h, patch_w = self.pair(self.patch_size)
        assert img_h % patch_h == 0 and img_w % patch_w == 0
        n_h = img_h // patch_h
        n_w = img_w // patch_w
        n_batch = x.size(0)
        patch_embed = x.view(n_batch, n_h*n_w, -1)
        dim = patch_embed.size(-1)
        cls_token = nn.Parameter(torch.randn(n_batch,1,dim))
        clsed_patch = torch.cat((cls_token,patch_embed),dim=1)
        pos_embed = transformer.PositionEmbedding(clsed_patch.size(-1),dropout = 0.1)
        clsed_patch =  pos_embed(clsed_patch)
        return clsed_patch
    

class Vit(nn.Module):
    def __init__(self, img_size, patch_size, n_class, dim, depth, heads, mlp_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_class = n_class
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
    
    def forward(self, x):
        image_process = ImagePreprocess(self.img_size,self.patch_size)
        processed_image = image_process(x)
        logger.debug('processed image size is {}'.format(str(processed_image.size())))
        transformer_part = TransformerEncoder(processed_image)
        output = transformer_part(processed_image,self.heads,self.mlp_dim,self.depth)
        generator_engine = generator(output.size(-1),self.n_class)
        output = generator_engine(output)
        logger.debug('output tensor size is {}'.format(str(output.size())))
        return output
















