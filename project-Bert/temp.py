import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)* -(math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x+self.pe[:,0:x.size(1)]#.clone().detach().requires_grad_(False)
    
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2,-1))\
            / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_layer = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q, K, V = [l(x).view(batch_size, self.num_heads, -1, self.d_k)
                   for l, x in zip(self.linear_layers, (Q, K, V))]
        # batch_size, num_heads, seq_len, d_k
        x, attn = self.attention(Q, K, V, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.d_k)
        return self.output_layer(x)
    
class ResConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        """
        :param d_model: transformer的隐藏层大小
        :param num_heads: 多头注意力的头数
        :param dff: 前馈网络的隐藏层大小，通常是4*d_model
        :param dropout: dropout率
        """
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.input_sublayer = ResConnection(d_model, dropout)
        self.output_sublayer = ResConnection(d_model, dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention(_x,_x,_x,mask=mask))
        x = self.output_sublayer(x, self.ffn)
        return self.dropout(x)
    
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=6, num_heads=12, dff=3072, max_len=512, dropout=0.1):
        """
        :param vocab_size: 词汇表的大小。
        :param d_model: BERT模型的隐藏层大小，默认为768。
        :param num_layers: Transformer块（层）的数量，默认为12。
        :param num_heads: 注意力头的数量，默认为12。
        :param dff: 前馈网络的隐藏层大小，默认为3072。
        :param max_len: 输入序列的最大长度，默认为512。
        :param dropout: dropout率，默认为0.1。
        """
        super(BERT, self).__init__()
        self.hidden_size = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = 4*d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)
             for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        if mask is None:
            mask = (x>0).unsqueeze(1).repeat(1,x.size(1),1).unsqueeze(1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, mask=mask)
        return x
    
import bertlib
from bertlib import MultiHeadAttention, FeedForward, ResConnection, PositionalEncoding
import torch.nn as nn
import torch

class DeconderLayer(nn.Module):
    def __init__(self, d_model=768, num_heads=6, dff=3072, dropout=0.1):
        super(DeconderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dec_enc_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = FeedForward(d_model, dff, dropout)
        self.sublayer1 = ResConnection(d_model, dropout)
        self.sublayer2 = ResConnection(d_model, dropout)
        self.sublayer3 = ResConnection(d_model, dropout)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, 
                dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        dec_self_attn_ouputs = self.sublayer1(dec_inputs,
                        lambda x: self.dec_self_attn(x, x, x, dec_self_attn_mask))
        dec_enc_attn_outputs = self.sublayer2(dec_self_attn_ouputs,
                        lambda x: self.dec_enc_attn(x, enc_outputs, enc_outputs, dec_enc_attn_mask))
        dec_outputs = self.sublayer3(dec_enc_attn_outputs, self.ffn)

        return dec_outputs
def create_self_mask(dec_inputs):
    """
    [0] : “PAD”
    """
    batch_size, tgt_len = dec_inputs.shape
    mask = torch.not_equal(dec_inputs,0).int()
    mask = mask.unsqueeze(1).repeat(1,tgt_len,1)
    look_ahead = torch.tril(torch.ones((tgt_len,tgt_len)))
    mask = mask * look_ahead
    return mask

def create_cross_mask(enc_inputs, dec_inputs):
    batch_size, src_len = enc_inputs.shape
    _, tgt_len = dec_inputs.shape
    mask = torch.not_equal(enc_inputs,0).int()
    mask = mask.unsqueeze(1).repeat(1,tgt_len,1)
    dec_mask = torch.not_equal(dec_inputs,0).int()
    dec_mask = dec_mask.unsqueeze(1).repeat(1,src_len,1)
    cross_mask = dec_mask.permute(0,2,1)*mask
    return cross_mask

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model=768, num_layers=6, num_heads=6, dff=3072, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1024)
        self.layers = nn.ModuleList([DeconderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])

    def forward(self, dec_inputs, enc_outputs, enc_inputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = self.embedding(dec_inputs)
        dec_outputs = self.pos_encoding(dec_outputs)
        dec_self_attn_mask = create_self_mask(dec_inputs)
        dec_enc_attn_mask = create_cross_mask(enc_inputs, dec_inputs)
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=768,
                 num_layers=6, num_heads=6, dff=3072, max_len=1024, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = bertlib.BERT(src_vocab_size, d_model, num_layers, num_heads, dff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, dff, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, enc_inputs, dec_inputs):
        enc_ouputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_ouputs, enc_inputs)
        logits = self.linear(dec_outputs)
        return logits