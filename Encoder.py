import torch.nn as nn
from MultiHeadAttenton import MultiHeadAttention
from PositionalWiseFeedForward import PositionalWiseFeedForward
from PositionalEncoding import PositionalEncoding
from Utils import *
from Elmo import Elmo

class EncoderLayer(nn.Module):

    def __init__(self,
                 d_model = 512,
                 num_heads = 8,
                 ffn_dim = 2018,
                 dropout = 0.0):

        super(EncoderLayer,self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, attn_mask = None):

        context = self.attention(x,x,x,attn_mask)
        output = self.feed_forward(context)

        return output

class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers = 6,
                 d_model = 512,
                 num_heads = 8,
                 ffn_dim = 2048,
                 dropout = 0.0):

        super(Encoder,self).__init__()

        self.encoder_layers = nn.ModuleList(
                            [EncoderLayer(d_model,num_heads,ffn_dim,dropout) for _ in range(num_layers)])

        self.elmo = Elmo(max_seq_len, num_layers)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len,dropout)

    def forward(self, x, seq_embedding):

        embedding = seq_embedding(x)
        output = self.pos_embedding(embedding)
        statelist = [output]

        self_attention_mask = padding_mask(x,x)

        for encoder in self.encoder_layers:
            output = encoder(output,self_attention_mask)
            statelist.append(output)

        output = self.elmo(statelist)

        batch_size, max_len, d_model = output.shape
        mu = torch.mean(output, dim=2).reshape(batch_size, max_len, -1)
        std = torch.std(output, dim=2).reshape(batch_size, max_len, -1)
        output = (output - mu) / std

        return output

