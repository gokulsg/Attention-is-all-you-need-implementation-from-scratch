import torch
import torch.nn as nn
import math

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding size needs to be divisible by number of heads"
        
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.keys = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, q, k, v, mask = None):
        
        batch_size = q.shape[0]
        
        #q = [batch size, q len, embed dim]
        #k = [batch size, k len, embed dim]
        #v = [batch size, v len, embed dim]
                
        query = self.queries(q)
        key = self.keys(k)
        value = self.values(v)
        
        #query = [batch size, query len, embed dim]
        #key = [batch size, key len, embed dim]
        #value = [batch size, value len, embed dim]

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        #query = [batch size, num heads, query len, head dim]
        #key = [batch size, num heads, key len, head dim]
        #value = [batch size, num heads, value len, head dim]
                
        q_bmm_kt = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim) 
        
        #q_bmm_kt = [batch size, num heads, query len, key len]
        
        if mask is not None:
            q_bmm_kt = q_bmm_kt.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(q_bmm_kt, dim = -1)
                
        #attention = [batch size, num heads, query len, key len]
                
        out = torch.matmul(attention, value)
        
        #out = [batch size, num heads, query len, head dim]
        
        out = out.transpose(1,2).reshape(batch_size, -1, self.embed_dim)
        
        #out = [batch size, query len, embed dim]
        
        out = self.fc_out(out)
        
        #out = [batch size, query len, embed dim]
        
        return out