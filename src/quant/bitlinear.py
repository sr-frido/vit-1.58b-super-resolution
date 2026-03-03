import torch
import torch.nn as nn
import torch.nn.functional as F


def round_clip(x, a=-1.0, b=1.0):
    return torch.clamp(torch.round(x), a, b)


def ternary_absmean_quant(w: torch.Tensor, alpha: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    gamma = w.abs().mean().clamp_min(eps)
    wq = round_clip(w / gamma, -1.0, 1.0)  # {-1,0,1}

    # mezcla: alpha=0 -> FP, alpha=1 -> ternario
    w_mix = (1.0 - alpha) * w + alpha * (wq * gamma)

    # STE (para la parte cuantizada)
    return w + (w_mix - w).detach()


def absmax_act_quant(x: torch.Tensor, bits: int = 8, eps: float = 1e-8) -> torch.Tensor:
    """
    Cuantización de activaciones basada en absmax (per-tensor), simulada en float.
    STE: forward usa cuantizado, backward identidad.
    """
    Qb = 2 ** (bits - 1)  # 128 para bits=8
    gamma = x.abs().max().clamp_min(eps)
    x_scaled = x * (Qb / gamma)
    x_clipped = torch.clamp(x_scaled, -Qb + eps, Qb - eps)

    # opcional: simular niveles discretos (redondeo). Esto puede ser más agresivo.
    # xq = torch.round(x_clipped)
    xq = x_clipped

    return x + (xq - x).detach()

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, act_bits=8):
        super().__init__()  # <-- imprescindible
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.act_bits = act_bits

        self.quant_act = False
        self.quant_w = False

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def set_quant(self, *, act=None, w=None, alpha_w=None):
        if act is not None:
            self.quant_act = act
        if w is not None:
            self.quant_w = w
        # alpha_w lo ignoramos aquí si aún no lo estás usando

    def forward(self, x):
        x = F.layer_norm(x, x.shape[-1:])

        if self.quant_act:
            x = absmax_act_quant(x, bits=self.act_bits)

        w = ternary_absmean_quant(self.weight) if self.quant_w else self.weight
        return F.linear(x, w, self.bias)