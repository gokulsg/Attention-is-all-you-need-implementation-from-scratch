import torch
import torch.nn as nn
from MultiHeadAttentionLayer import MultiHeadAttentionLayer
from PositionwiseFeedforwardLayer import PositionwiseFeedforwardLayer

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, expand_dim, dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(embed_dim, expand_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, embed dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        attn = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.dropout(self.self_attn_layer_norm(src + attn)) # --> make correction the dropout should be used only on self attention
        
        #src = [batch size, src len, embed dim]
        
        #positionwise feedforward
        x = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        out = self.feed_forward_layer_norm(src + self.dropout(x))
        
        #src = [batch size, src len, embed dim]
        
        return out