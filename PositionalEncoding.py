import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):

    #compute position encoding

    def __init__(self,
                 d_model,
                 max_seq_len,
                 dropout=0.0):

        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len,d_model)
        position = torch.arange(0.,max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.,d_model,2)*-(math.log(10000.0)/d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)


    def forward(self,x):

        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)

        return self.dropout(x)



