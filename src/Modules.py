import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN as OriginalKANLinear
from typing import Any, Dict
from enum import Enum
T = torch.FloatTensor

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, similarity, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn + similarity.to('cuda'), dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

    
class ModelType(str, Enum):
    MLP = "mlp"
    KAN_ORIGINAL = "kan_original"
    KAN_EFFICIENT = "kan_efficient"
    KAN_CHEBYSHEV = "kan_chebyshev"
    KAN_FAST = "kan_fast"

def get_linear_cls( model_type: ModelType,use_kan_bias: bool, *, chebykan_degree: int = 4, **kwargs: Dict[str, Any],) -> nn.Module:  
    if model_type == ModelType.MLP:   
        return lambda in_features, out_features, bias: nn.Linear(
            in_features, out_features, bias=bias
        )
    elif model_type == ModelType.KAN_ORIGINAL:
        return lambda in_features, out_features, bias: KANOriginal(
            in_features, out_features
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
   

class KANOriginal(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.kan = OriginalKANLinear(
            width=[in_features, out_features],
            grid=3,
            k=3,
            noise_scale=0.1,
            noise_scale_base=0.1,
            base_fun=torch.nn.SiLU,
            symbolic_enabled=True,
            bias_trainable=True,
            grid_eps=1.0,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device='cuda'
        )

    def forward(self, x: T) -> T:
        if x.device != self.kan.device:
            self.kan = self.kan.to(x.device)
            self.kan.device = x.device
            for layer in self.kan.act_fun:
                layer.to(x.device)
                layer.device = x.device
        batch_size, seq_length, _ = x.shape
        x = self.kan(x.view(-1, x.shape[-1]))
        return x.view(batch_size, seq_length, -1)

