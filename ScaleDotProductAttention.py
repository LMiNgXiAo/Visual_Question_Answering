import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):

    def __init__(self,
                 attention_dropout=0.0):

        super(ScaledDotProductAttention,self).__init__()

        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim = -1)


    def forward(self,q,k,v,scale=None,attn_mask = None):

        attention = torch.matmul(q,k.transpose(-2,-1))

        if scale:
            attention = attention * scale

        # mask attention. The attentions between the masked words and
        # other words are set to negative infinity
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask,-np.inf)

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.matmul(attention,v)

        return context
