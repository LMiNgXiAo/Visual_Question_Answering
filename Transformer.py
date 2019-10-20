import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from Generator2 import Generator
from Utils import padding_mask


class Transformer(nn.Module):
    #Build transformer model

    def __init__(self,
                 vocab_size,
                 max_len,
                 device,
                 num_layers = 6,
                 stack_layers= 6,
                 d_model = 512,
                 num_heads = 8,
                 ffn_dim = 2048,
                 dropout = 0.2):

        super(Transformer, self).__init__()

        self.device = device

        self.encoder = Encoder(vocab_size, max_len,num_layers,d_model,num_heads,ffn_dim,dropout)
        self.decoder = Decoder(vocab_size, max_len,device, num_layers,d_model,num_heads, ffn_dim, dropout)
        self.generator = Generator(vocab_size,max_len,device,stack_layers,d_model,num_heads,ffn_dim,dropout)

        self.embedding = nn.Embedding(vocab_size,d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias = False)
        self.softmax = nn.Softmax(dim = 2)


    def forward(self, src_seq, dec_tgt,image_in):                           #

        context_attn_mask_dec = padding_mask(dec_tgt, src_seq)

        en_output = self.encoder(src_seq,self.embedding)

        dec_output = self.decoder(dec_tgt, en_output,self.embedding,context_attn_mask_dec)

        gen_output = self.generator(image_in,en_output)

        return dec_output, gen_output



