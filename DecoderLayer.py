import torch
import torch.nn as nn
from MultiHeadAttentionLayer import MultiHeadAttentionLayer
from PositionwiseFeedforwardLayer import PositionwiseFeedforwardLayer

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, expand_dim, dropout):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.encoder_attention = MultiHeadAttentionLayer(embed_dim, num_heads)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(embed_dim, expand_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, embed dim]
        #enc_src = [batch size, src len, embed dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        trg_attn = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.dropout(self.self_attn_layer_norm(trg + trg_attn))
            
        #trg = [batch size, trg len, embed dim]
            
        #encoder attention
        trg_attn = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.dropout(self.enc_attn_layer_norm(trg + trg_attn))
                    
        #trg = [batch size, trg len, embed dim]
        
        #positionwise feedforward
        x = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.feed_forward_layer_norm(trg + self.dropout(x))
        
        #trg = [batch size, trg len, embed dim]
        
        return trg