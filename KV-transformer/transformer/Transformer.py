import Bert
from Bert import MultiHeadAttention, FeedForward, ResConnection, PositionalEncoding
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
                dec_enc_attn_mask, cache=None):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        if cache is not None:
            dec_self_attn_ouputs = self.sublayer1(dec_inputs,
                            lambda x: self.dec_self_attn(x, x, x, dec_self_attn_mask, cache["self_attn"],update_KV=True))
            if cache["enc_attn"][0] is None: #KV cache is empty
                dec_enc_attn_outputs = self.sublayer2(dec_self_attn_ouputs,
                            lambda x: self.dec_enc_attn(x, enc_outputs, enc_outputs, dec_enc_attn_mask, cache["enc_attn"],update_KV=True))
            else:
                dec_enc_attn_outputs = self.sublayer2(dec_self_attn_ouputs,
                            lambda x: self.dec_enc_attn(x, enc_outputs, enc_outputs, dec_enc_attn_mask, cache["enc_attn"],update_KV=False))
        else:
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
    return mask.unsqueeze_(1)

def create_cross_mask(enc_inputs, dec_inputs):
    batch_size, src_len = enc_inputs.shape
    _, tgt_len = dec_inputs.shape
    mask = torch.not_equal(enc_inputs,0).int()
    mask = mask.unsqueeze(1).repeat(1,tgt_len,1)
    dec_mask = torch.not_equal(dec_inputs,0).int()
    dec_mask = dec_mask.unsqueeze(1).repeat(1,src_len,1)
    cross_mask = dec_mask.permute(0,2,1)*mask
    return cross_mask.unsqueeze_(1)

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model=768, num_layers=6, num_heads=6, dff=3072, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1024)
        self.layers = nn.ModuleList([DeconderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])

    def forward(self, dec_inputs, enc_outputs, enc_inputs, cache=None):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = self.embedding(dec_inputs)
        dec_outputs = self.pos_encoding(dec_outputs)
        dec_self_attn_mask = create_self_mask(dec_inputs)
        dec_enc_attn_mask = create_cross_mask(enc_inputs, dec_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attn_mask: [batch_size, 1, tgt_len, tgt_len],
        # dec_enc_attn_mask: [batch_size, 1, tgt_len, src_len]
        for i, layer in enumerate(self.layers):
            if cache is not None:
                if cache[i]["self_attn"][0] is not None:
                    dec_outputs = layer(dec_outputs[:,-1:,:], enc_outputs, None, dec_enc_attn_mask,cache=cache[i]) # no mask for self-attention in decoder
                else:
                    dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask,cache=cache[i])
            else:
                dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=768,
                 num_layers=6, num_heads=6, dff=3072, max_len=1024, dropout=0.1):#,using_cache=False, max_cache_len=512):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.initial_cache()
        #self.max_cache_len = max_cache_len
        self.encoder = bertlib.BERT(src_vocab_size, d_model, num_layers, num_heads, dff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, dff, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    def initial_cache(self, using_cache=False, max_cache_len=512):
        if using_cache:
            self.cache= [
                {"self_attn":[
                    None, None
                ],
                "enc_attn":[
                    None, None
                ]
                }
                for _ in range(self.num_layers)]
            self.max_cache_len = max_cache_len
        else:
            self.cache = None

    def forward(self, enc_inputs, dec_inputs):
        if self.cache is None:
            enc_ouputs = self.encoder(enc_inputs)
            dec_outputs = self.decoder(dec_inputs, enc_ouputs, enc_inputs)
        elif self.cache[0]["enc_attn"][0] is None: #KV cache is empty
            #print("cache is empty")
            enc_ouputs = self.encoder(enc_inputs)
            dec_outputs = self.decoder(dec_inputs, enc_ouputs, enc_inputs,self.cache)
        else:
            dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_inputs,self.cache) #the second parameter enc_inputs is not used
        
        self.manage_cache()  # manage cache size automatically
        logits = self.linear(dec_outputs)
        return logits, dec_outputs
    
    def manage_cache(self):
        if self.cache is not None:
            for i in range(len(self.cache)):
                key = ["self_attn", "enc_attn"]
                for j in range(len(key)):
                    if self.cache[i][key[j]][0] is not None:
                        if len(self.cache[i][key[j]][0]) > self.max_cache_len:
                            self.cache[i][key[j]][0] = self.cache[i][key[j]][0][:,:,-self.max_cache_len:,:]
                            self.cache[i][key[j]][1] = self.cache[i][key[j]][1][:,:,-self.max_cache_len:,:]
    
    def clear_cache(self):
        if self.cache is not None:
            for i in range(len(self.cache)):
                self.cache[i]["self_attn"] = [None, None]
                self.cache[i]["enc_attn"] = [None, None]
    
    def remove_cache(self):
        self.cache = None