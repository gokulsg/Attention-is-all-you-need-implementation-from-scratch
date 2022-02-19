import torch
import torch.nn as nn

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, embed_dim, expand_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(embed_dim, expand_dim)
        self.fc_2 = nn.Linear(expand_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, seq len, embed dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        #x = [batch size, seq len, expand dim]

        x = self.fc_2(x)

        #x = [batch size, seq len, embed dim]

        return x