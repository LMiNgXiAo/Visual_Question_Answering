import torch.nn.functional as F
import torch.nn as nn

class PositionalWiseFeedForward(nn.Module):

    def __init__(self,
                 d_model=512,
                 ffn_dim=2048,
                 dropout=0.0):

        super(PositionalWiseFeedForward,self).__init__()

        self.w1 = nn.Linear(d_model,ffn_dim)
        self.w2 = nn.Linear(ffn_dim,d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)


    def forward(self,x):

        output = self.w2(F.relu(self.w1(x)))
        # layer normalization and residual network
        return self.norm(x+self.dropout(output))

