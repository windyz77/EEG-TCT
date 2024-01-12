import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Reduce

d_ff = 2048  # FeedForward dimension (62-256-62线性提取的过程)
d_k = 128
d_v = 128  # dimension of K(=Q), V
n_heads = 16  # number of heads in Multi-Head Attention

# Transformer ScaledDot_Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self,):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# Transformer Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1,  n_heads,  d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1,  n_heads,  d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1,  n_heads,  d_v).transpose(1,
                                                                           2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1,  n_heads, 1,
                                                      1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads *  d_v)
        output = self.fc(context)
        d_model = output.shape[2]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


# Transformer PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU()
        )
        self.batchNorm = nn.BatchNorm1d(d_ff)  # ??? xmy
        self.fc2 = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        input_fc1 = self.fc(inputs)
        input_fc1 = input_fc1.permute(0, 2, 1)
        input_bn = self.batchNorm(input_fc1)
        input_bn = input_bn.permute(0, 2, 1)

        output = self.fc2(input_bn)
        d_model = output.shape[2]
        return nn.LayerNorm(d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)
        self.pos_ffn = PoswiseFeedForwardNet(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        # layer residual
        enc_outputs = enc_outputs + enc_inputs
        return enc_outputs, attn


# Transformer Encoder on time dimension
class Time_Encoder(nn.Module):
    def __init__(self,channel_size,time_size,n_layers):
        super(Time_Encoder, self).__init__()
        self.pos_emb = nn.Parameter(torch.randn(1,time_size , channel_size))
        self.dropout = nn.Dropout(0.6)
        self.layers = nn.ModuleList([EncoderLayer(d_model=channel_size) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = enc_inputs
        b, n, _ = enc_outputs.shape

        # Position Embedding
        enc_outputs += self.pos_emb[:, :n]
        # record position embedding
        enc_pos_output = enc_outputs
        enc_outputs = self.dropout(enc_outputs)


        enc_self_attn_mask = None
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        # output + pos_output
        enc_outputs = enc_outputs + enc_pos_output
        return enc_outputs, enc_self_attns

# Transformer Encoder on channel dimension
class Channel_Encoder(nn.Module):
    def __init__(self,channel_size,time_size,d_model,tokens,n_layers):
        super(Channel_Encoder, self).__init__()
        self.src_emb = nn.Linear(time_size, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(1, channel_size+tokens, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, tokens, d_model))
        self.dropout = nn.Dropout(0.6)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model) for _ in range(n_layers)])
        self.tokens = tokens
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs = enc_inputs
        # cls_token
        b, n, _ = enc_outputs.shape
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        enc_outputs = torch.cat((cls_tokens, enc_outputs), dim=1)

        # Position embedding
        enc_outputs += self.pos_emb[:, :(n + self.tokens)]
        # record position embedding
        enc_pos_output = enc_outputs
        enc_outputs = self.dropout(enc_outputs)


        enc_self_attn_mask = None
        enc_self_attns = []

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        # output + pos_output
        enc_outputs = enc_outputs + enc_pos_output
        return enc_outputs, enc_self_attns

# Classification
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size * 2),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(emb_size * 2, emb_size),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class Transformer(nn.Module):
    def __init__(self,config):
        super(Transformer, self).__init__()
        self.time_encoder = Time_Encoder(channel_size=config.channel_size,time_size=config.time_size,n_layers=config.n_layers).cuda()
        self.channel_encoder = Channel_Encoder(channel_size=config.channel_size,time_size=config.time_size,d_model=config.d_model,tokens=config.tokens,n_layers=config.n_layers).cuda()
        self.classification = ClassificationHead(emb_size=config.d_model,n_classes= config.n_classes)
        self.tokens = config.tokens

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        dec_inputs: [batch_size, tgt_len, d_model]
        '''

        # Time Transformer [8, 201, 192]->[8, 201, 192]
        enc_inputs_time = enc_inputs
        enc_time_outputs, enc_self_attns = self.time_encoder(enc_inputs_time)

        # Channel transformer [8, 192, 201]->[8, 196, 256]
        # enc_channel_inputs = enc_time_outputs + enc_inputs
        enc_channel_inputs = enc_time_outputs
        enc_channel_inputs = rearrange(enc_channel_inputs, 'b t c->b c t')
        enc_outputs_channel, enc_self_attns = self.channel_encoder(enc_channel_inputs)

        # Classification [8, 196, 256]->[8, 4, 256]->[8, 26]
        cls_output_channel = enc_outputs_channel[:, :self.tokens]
        classres_output = self.classification(cls_output_channel)
        return classres_output, enc_self_attns

