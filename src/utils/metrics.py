import torch

def psnr(sr, hr, eps=1e-10):
    # sr, hr: [N,C,H,W] in [0,1]
    mse = torch.mean((sr - hr) ** 2, dim=(1,2,3))
    return 10.0 * torch.log10(1.0 / (mse + eps))