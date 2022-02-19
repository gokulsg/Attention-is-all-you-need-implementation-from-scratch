import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer
import math

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, expand_dim, dropout, device, max_length = 30):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, embed_dim)
        #self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(self.get_positional_encoding(max_length, embed_dim))
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, expand_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
        self.device = device
        
    def forward(self, src, src_mask):
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, embed dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, embed dim]
            
        return src
    
    def get_positional_encoding(self, max_seq_len, embed_dim):
        pos_enc = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc