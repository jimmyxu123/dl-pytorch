import torch
import transformer
from transformer import *
import torch.nn as nn
import copy
from loguru import logger

class TransformerEncoder(nn.Module):
    def __init__(self,heads, mlp_dim, depth):
        super().__init__()
        self.dropout = 0.1
        self.depth = depth
        self.d=3072
        self.attn = transformer.MultiHeadAttention(heads,self.d)
        self.ff = transformer.PositionWiseFeedForward(self.d,mlp_dim,self.dropout)
        self.encoder_layer = transformer.EncoderLayer(self.d,self.attn,self.ff,self.dropout)
        self.encode = transformer.Encoder(self.encoder_layer, self.depth)
        
    def forward(self,input):
        output = self.encode(input, mask=None)
        return output

class generator(nn.Module):
    def __init__(self, d_model, n_class):
        super().__init__()
        self.n_class = n_class
        self.mlp = nn.Sequential(nn.LayerNorm(d_model),
                            nn.Linear(d_model,self.n_class))

    def forward(self,x):
        x = x.mean(dim=1)
        return F.log_softmax(self.mlp(x),dim=-1)

class ImagePreprocess(nn.Module):
    def __init__(self, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        img_h, img_w = self.pair(self.img_size)
        patch_h, patch_w = self.pair(self.patch_size)
        assert img_h % patch_h == 0 and img_w % patch_w == 0
        self.n_h = img_h // patch_h
        self.n_w = img_w // patch_w
        self.pos_embed = transformer.PositionEmbedding(3072,dropout = 0.1)
    
    def pair(self,x):
        return x if isinstance(x, tuple) else (x,x)
    
    def forward(self,x):
        n_batch = x.size(0)
        patch_embed = x.view(n_batch, self.n_h*self.n_w, -1)
        dim = patch_embed.size(-1)
        cls_token = torch.randn(n_batch,1,dim)
        clsed_patch = torch.cat((cls_token,patch_embed),dim=1)
        self.clsed_patch =  self.pos_embed(clsed_patch)   
        return self.clsed_patch
    

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
        self.image_process = ImagePreprocess(self.img_size,self.patch_size) 
        self.transformer_part = TransformerEncoder(self.heads,self.mlp_dim,self.depth)
        self.generator_engine = generator(3072,self.n_class)
   
    
    def forward(self, x):
        #self.transformer_part = TransformerEncoder(x,self.heads,self.mlp_dim,self.depth)
        processed_image = self.image_process(x)
        logger.debug('processed image size is {}'.format(str(processed_image.size())))
        output = self.transformer_part(processed_image)
        output = self.generator_engine(output)
        logger.debug('output tensor size is {}'.format(str(output.size())))
        return output
















