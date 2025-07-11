import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.Constants as Constants
from src.Modules import ScaledDotProductAttention, get_linear_cls


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v,model_type, use_bias, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        if model_type=='mlp':
            self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
            nn.init.xavier_uniform_(self.w_qs.weight)
            nn.init.xavier_uniform_(self.w_ks.weight)
            nn.init.xavier_uniform_(self.w_vs.weight)

            self.fc = nn.Linear(d_v * n_head, d_model)
            nn.init.xavier_uniform_(self.fc.weight)
        else:
            cls = get_linear_cls(model_type, use_bias)
            self.w_qs = cls(d_model, n_head * d_k, bias=False)
            self.w_ks = cls(d_model, n_head * d_k, bias=False)
            self.w_vs = cls(d_model, n_head * d_v, bias=False)
            
            self.fc = cls(d_v * n_head, d_model,bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, similarity, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q)
        q=q.view(sz_b,len_q,d_k,-1)

        k = self.w_ks(k)
        k=k.view(sz_b,len_q,d_k,-1)

        v = self.w_vs(v)
        v=v.view(sz_b,len_q,d_k,-1)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        output, attn = self.attention(q, k, v, similarity.to("cuda") , mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid,model_type, use_bias, dropout=0.1, normalize_before=True, **kwargs):
        super().__init__()
        self.normalize_before = normalize_before

        if model_type=='mlp':
            self.w_1 = nn.Linear(d_in, d_hid)
            self.w_2 = nn.Linear(d_hid, d_in)
        else:
            cls = get_linear_cls(model_type, use_bias, **kwargs)
            self.w_1 = cls(d_in, d_hid, bias=use_bias)
            self.w_2 = cls(d_hid, d_in,bias=use_bias)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
