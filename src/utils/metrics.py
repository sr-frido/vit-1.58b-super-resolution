import torch
import torch.nn.functional as F


def _shave(x: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return x
    return x[..., shave:-shave, shave:-shave]


def rgb_to_y(sr: torch.Tensor) -> torch.Tensor:
    """
    Convierte RGB [0,1] a canal Y [0,1] usando coeficientes tipo BT.601.
    Entrada: [N,3,H,W]
    Salida:  [N,1,H,W]
    """
    if sr.size(1) != 3:
        raise ValueError(f"rgb_to_y espera 3 canales, recibió {sr.size(1)}")

    r = sr[:, 0:1, :, :]
    g = sr[:, 1:2, :, :]
    b = sr[:, 2:3, :, :]

    # Y en rango [0,1]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def psnr_rgb(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    PSNR sobre full RGB.
    sr, hr: [N,C,H,W] en [0,1]
    Devuelve: [N]
    """
    mse = torch.mean((sr - hr) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def psnr_y(sr: torch.Tensor, hr: torch.Tensor, shave: int = 4, eps: float = 1e-10) -> torch.Tensor:
    """
    PSNR sobre canal Y (luminancia), con recorte de bordes.
    sr, hr: [N,3,H,W] en [0,1]
    Devuelve: [N]
    """
    sr_y = rgb_to_y(sr)
    hr_y = rgb_to_y(hr)

    sr_y = _shave(sr_y, shave)
    hr_y = _shave(hr_y, shave)

    mse = torch.mean((sr_y - hr_y) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, window_size, window_size)


def _ssim_single_channel(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5):
    """
    x, y: [N,1,H,W] en [0,1]
    Devuelve: [N]
    """
    device = x.device
    dtype = x.dtype
    kernel = _gaussian_kernel(window_size, sigma, device=device, dtype=dtype)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.conv2d(x, kernel, padding=window_size // 2)
    mu_y = F.conv2d(y, kernel, padding=window_size // 2)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    return ssim_map.mean(dim=(1, 2, 3))


def ssim_rgb(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """
    SSIM promedio sobre los 3 canales RGB.
    sr, hr: [N,3,H,W] en [0,1]
    Devuelve: [N]
    """
    vals = []
    for c in range(3):
        vals.append(_ssim_single_channel(sr[:, c:c+1], hr[:, c:c+1]))
    return torch.stack(vals, dim=0).mean(dim=0)


def ssim_y(sr: torch.Tensor, hr: torch.Tensor, shave: int = 4) -> torch.Tensor:
    """
    SSIM sobre canal Y con recorte de bordes.
    sr, hr: [N,3,H,W] en [0,1]
    Devuelve: [N]
    """
    sr_y = _shave(rgb_to_y(sr), shave)
    hr_y = _shave(rgb_to_y(hr), shave)
    return _ssim_single_channel(sr_y, hr_y)


class LPIPSMetric:
    """
    Wrapper opcional para LPIPS.
    Requiere: pip install lpips
    Uso:
        lpips_metric = LPIPSMetric(net='alex', device='cuda')
        val = lpips_metric(sr, hr)   # devuelve [N]
    """
    def __init__(self, net: str = "alex", device: str = "cpu"):
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "LPIPS no está instalado. Ejecuta: pip install lpips"
            ) from e

        self.device = device
        self.metric = lpips.LPIPS(net=net).to(device)
        self.metric.eval()

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """
        LPIPS espera rango [-1,1].
        Entrada aquí: [N,3,H,W] en [0,1]
        Devuelve: [N]
        """
        sr_n = sr * 2.0 - 1.0
        hr_n = hr * 2.0 - 1.0
        out = self.metric(sr_n, hr_n)
        return out.view(-1)