import torch.nn as nn
from MultiHeadAttenton import MultiHeadAttention
from PositionalWiseFeedForward import PositionalWiseFeedForward
import torch

class GeneratorLayer(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads = 8,
                 ffn_dim = 2048,
                 dropout = 0.0):

        super(GeneratorLayer,self).__init__()

        self.attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model,ffn_dim,dropout)

    def forward(self, img_in,ques_in,self_attn_mask = None, context_attn_mask=None):

        output = self.attention(img_in,img_in,img_in,self_attn_mask)

        output = self.attention(ques_in,ques_in,output,context_attn_mask)

        output = self.feed_forward(output)

        return output

class Generator(nn.Module):

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 device,
                 num_layers = 3,
                 d_model = 512,
                 num_heads = 8,
                 ffn_dim = 1024,
                 dropout =0.0):

        super(Generator,self).__init__()
        self.device = device
        self.num_layers = num_layers

        self.generator_layers = nn.ModuleList(
            [GeneratorLayer(d_model,num_heads,ffn_dim,dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(196,max_seq_len)
        self.linear2 = nn.Linear(d_model,vocab_size,bias=False)


    def forward(self,img_in,enc_output):

        batch_size, num_fea, fea = img_in.shape
        mu = torch.mean(img_in, dim=2).reshape(batch_size, num_fea, -1)
        std = torch.std(img_in, dim=2).reshape(batch_size, num_fea, -1)
        img_in = (img_in - mu) / std

        img_in = self.linear(img_in.permute(0,2,1))
        img_in = img_in.permute(0,2,1)


        for generator in self.generator_layers:
            output = generator(img_in, enc_output)

        output = self.linear2(output)

        return output
