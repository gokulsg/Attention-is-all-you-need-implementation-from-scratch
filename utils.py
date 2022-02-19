import torch
import torch.nn as nn

# counting Number of parameters in a model #
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## weight initialization ##
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

