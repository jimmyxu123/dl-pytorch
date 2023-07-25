import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import numpy as np
from torch.autograd import Variable
#positon wise mlp
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(int(vocab.size(0)), self.d_model)
    
    def forward(self, x):
        out = self.emb(x) * math.sqrt(self.d_model)
        out = out.unsqueeze(0)
        return out


    
class PositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        #position: [max_len, 1]
        exp = torch.arange(0,d_model, 2)/d_model
        div_term = 1000 ** exp
        pe[:,0::2] = torch.sin(position / div_term)
        pe[:,1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1),:],
                         requires_grad = False)
#        return self.dropout(x)
        return x

class GeneralEmbedding(nn.Module):
    def __init__(self,d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
    
    def forward(self, x):
        eb_1 = Embedding(self.d_model, x)
        x_1 = eb_1(x)
        eb_2 = PositionEmbedding(self.d_model, self.dropout)
        x_2 = eb_2(x_1)
#        x_2 = torch.where(torch.isnan(x_2), torch.full_like(x_2, 1), x_2)
        return x_2.long()


#basic structure
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder,src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self,tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory,src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src,src_mask)
        decode = self.decode(tgt, memory,src_mask,tgt_mask)
        generator = Generator(decode.size(-1),src.size(-1))
        return generator(decode)

class Generator(nn.Module):
    def __init__(self, d_model, voc):
        super().__init__()
        self.proj = nn.Linear(d_model, voc)
    
    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(features)
    
    def forward(self,x):
        return self.layernorm(x)

class LayerNorm_(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.float().mean(-1, keepdim =True)
        std = x.float().std(-1, keepdim =True)
        return (self.a * (x-mean) / (std+self.eps)) + self.b

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, size, atten, feed_forward, dropout):
        super().__init__()
        self.attn = atten
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.size = size
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)
        return self.norm(tgt)
    
class DecoderLayer(nn.Module):
    def __init__(self, size, atten, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.attn = atten
        self.feed_forward = feed_forward
        self.src_attn = src_attn
        self.sublayer = clone(SublayerConnection(size, dropout),3)
    
    def forward(self, x, memory, src_msk, tgt_msk):
        m = memory
        x = self.sublayer[0](x, lambda x: self.attn(x,x,x,tgt_msk))
        x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_msk))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_mask =np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(q,k,v, mask = None, dropout = None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weight_matrix = F.softmax(scores, dim=-1)
    if dropout is not None:
        weight_matrix = dropout(weight_matrix)
    return torch.matmul(weight_matrix, v), weight_matrix

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model / self.h
        self.linears = clone(nn.Linear(d_model, d_model),3)
        self.attn = None
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, q,k,v,mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = q.size(0)
        q, k, v = [l(x).view(n_batch, -1, int(self.h), int(self.d_k)).transpose(1, 2)
             for l, x in zip(self.linears, (q, k, v))]
        #q,k,v = [l(x).view(n_batch, -1, int(self.h), int(self.d_k)).transpose(1,2) for x, l in zip((q,k,v),self.linears)]
        output, weight_matrix = attention(q,k,v,mask=mask, dropout=self.dropout)
        #weigth_matrix: (n_batch, self.h, number of words, number of words)
        #output: (n_batch, self.h, number of words, d_k)
        output = output.transpose(1,2).contiguous().view(n_batch, -1, int(self.h * self.d_k))
        d_model = int(output.size(-1))
        linear = nn.Linear(d_model, d_model)
        return linear(output)

def model(src,tgt,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h,d_model)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    encoder = Encoder(EncoderLayer(d_model, c(attn),c(ff),dropout),N)
    decoder = Decoder(DecoderLayer(d_model, c(attn),c(attn),c(ff),dropout),N)
    src_embed = GeneralEmbedding(d_model)
    tgt_embed = GeneralEmbedding(d_model)
    generator = Generator(d_model, tgt.size(0))#consider it as a weight matrix
    model = EncoderDecoder(encoder,decoder,src_embed,tgt_embed,generator)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    outp = model(src,tgt,src_mask = None, tgt_mask=None)
    return outp


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,src,tgt,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
        c = copy.deepcopy
        attn = MultiHeadAttention(h,d_model)
        ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        encoder = Encoder(EncoderLayer(d_model, c(attn),c(ff),dropout),N)
        decoder = Decoder(DecoderLayer(d_model, c(attn),c(attn),c(ff),dropout),N)
        src_embed = GeneralEmbedding(d_model)
        tgt_embed = GeneralEmbedding(d_model)
        generator = Generator(d_model, tgt.size(0))#consider it as a weight matrix
        model = EncoderDecoder(encoder,decoder,src_embed,tgt_embed,generator)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        outp = model(src,tgt,src_mask = None, tgt_mask=None)
        return outp







