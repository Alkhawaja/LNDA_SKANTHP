import torch.nn as nn

from src.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v,model_type, use_bias, dropout=0.1, normalize_before=True,):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, model_type, use_bias,dropout=dropout, normalize_before=normalize_before,)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner,model_type, use_bias, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, similarity, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, similarity, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
