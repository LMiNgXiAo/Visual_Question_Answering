import torch
import torch.nn as nn
from MultiHeadAttenton import MultiHeadAttention
from PositionalWiseFeedForward import PositionalWiseFeedForward
from PositionalEncoding import PositionalEncoding
from Utils import padding_mask,sequence_mask

class DecoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 num_heads = 8,
                 ffn_dim = 2048,
                 dropout = 0.0):

        super(DecoderLayer,self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(d_model, ffn_dim, dropout)


    def forward(self, dec_inputs, enc_outputs, self_attn_mask = None,context_attn_mask = None):

        dec_ouput  = self.attention(dec_inputs, dec_inputs, dec_inputs ,self_attn_mask)

        dec_ouput = self.attention(enc_outputs, enc_outputs,dec_ouput, context_attn_mask)

        dec_ouput = self.feed_forward(dec_ouput)

        return dec_ouput

class Decoder(nn.Module):

    def __init__(self,
                vocab_size,
                 max_seq_len,
                 device,
                 num_layers = 6,
                 d_model  = 512,
                 num_heads = 8,
                 ffn_dim = 2048,
                 dropout = 0.0,
                 ):

        super(Decoder,self).__init__()
        self.device = device
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model,num_heads,ffn_dim,dropout) for _ in range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = PositionalEncoding(d_model, max_seq_len)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, inputs, enc_output, seq_embedding, context_attn_mask = None):

        embedding = seq_embedding(inputs)
        output =  embedding + self.pos_embedding(embedding)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs).to(self.device)
        self_attn_mask = torch.gt((self_attention_padding_mask+seq_mask), 0 )


        for decoder in self.decoder_layers:
            output = decoder(output, enc_output,self_attn_mask,context_attn_mask)

        output = self.linear(output)
        return output






