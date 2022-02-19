import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_len = trg.shape[1]
        batch = trg.shape[0]
        
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch, 1, trg_len, trg_len)
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, embed dim]
                
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        
        return output