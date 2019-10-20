import torch.nn as nn
from ScaleDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    # compute multi heads attention

    def __init__(self,
                 d_modl=512,
                 num_heads=8,
                 dropout=0.0):

        super(MultiHeadAttention,self).__init__()

        self.dim_per_head = d_modl // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(d_modl, d_modl)
        self.linear_v = nn.Linear(d_modl, d_modl)
        self.linear_q = nn.Linear(d_modl, d_modl)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(d_modl,d_modl)
        self.norm = nn.LayerNorm(d_modl)


    def forward(self, keys, values, queries, attn_mask=None):

        residual = queries
        batch_size = keys.size(0)
        #generate keys,values and queries from inputs
        keys = self.linear_k(keys)
        values = self.linear_v(values)
        queries = self.linear_q(queries)

        keys = keys.view(batch_size , -1, self.num_heads, self.dim_per_head).transpose(1,2)
        values = values.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1,2)
        queries = queries.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1,2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_heads,1,1)

        scale = (keys.size(-1)) ** -0.5

        context = self.dot_product_attention(queries,keys,values,scale,attn_mask)

        context = context.transpose(1,2).contiguous() \
                  .view(batch_size,-1,self.num_heads * self.dim_per_head)

        # layer normalization and residual network
        return self.norm(residual+self.linear_final(context))






