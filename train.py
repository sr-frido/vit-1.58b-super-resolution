import os, yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.datasets.div2k import DIV2KPairDataset
from src.utils.metrics import psnr_rgb, psnr_y, ssim_y
from src.models.vit_sr_158b import ViTSR158b


import torch.nn.functional as F

@torch.no_grad()
def tile_inference(model, lr, scale=4, tile=64, overlap=16, device="cuda"):
    """
    lr: (1,3,H,W) en LR.
    tile/overlap están en píxeles LR.
    Devuelve sr: (1,3,H*scale,W*scale)
    """
    model.eval()
    assert lr.dim() == 4 and lr.size(0) == 1, "tile_inference asume batch=1"
    _, _, H, W = lr.shape

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile must be > overlap")

    outH, outW = H * scale, W * scale
    sr_acc = torch.zeros((1, 3, outH, outW), device=device, dtype=torch.float32)
    wt_acc = torch.zeros((1, 1, outH, outW), device=device, dtype=torch.float32)

    # recorre tiles
    y = 0
    while y < H:
        x = 0
        y0 = min(y, H - tile)
        y1 = y0 + tile
        while x < W:
            x0 = min(x, W - tile)
            x1 = x0 + tile

            lr_tile = lr[:, :, y0:y1, x0:x1]  # (1,3,tile,tile)
            sr_tile = model(lr_tile).clamp(0, 1)  # (1,3,tile*s,tile*s)

            oy0, oy1 = y0 * scale, y1 * scale
            ox0, ox1 = x0 * scale, x1 * scale

            sr_acc[:, :, oy0:oy1, ox0:ox1] += sr_tile.float()
            wt_acc[:, :, oy0:oy1, ox0:ox1] += 1.0

            x += stride
        y += stride

    sr = sr_acc / wt_acc.clamp_min(1e-8)
    return sr


def set_model_quant_mode(model, *, act: bool, w: bool, alpha_w: float = 1.0):
    for m in model.modules():
        if hasattr(m, "set_quant"):
            m.set_quant(act=act, w=w, alpha_w=alpha_w)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

#PSNR FULLRGB Y LUMINANCIA Y SSIM LUMINANCIA
@torch.no_grad()
def validate(model, dl, device, scale=4, tile=96, overlap=8):
    model.eval()
    psnr_rgb_vals = []
    psnr_y_vals = []
    ssim_y_vals = []

    for lr, hr in dl:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = tile_inference(model, lr, scale=scale, tile=tile, overlap=overlap, device=device).clamp(0, 1)

        psnr_rgb_vals.append(psnr_rgb(sr, hr).mean().item())
        psnr_y_vals.append(psnr_y(sr, hr, shave=scale).mean().item())
        ssim_y_vals.append(ssim_y(sr, hr, shave=scale).mean().item())

    return {
        "psnr_rgb": sum(psnr_rgb_vals) / max(1, len(psnr_rgb_vals)),
        "psnr_y": sum(psnr_y_vals) / max(1, len(psnr_y_vals)),
        "ssim_y": sum(ssim_y_vals) / max(1, len(ssim_y_vals)),
    }


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    exp = cfg["exp_name"]
    os.makedirs(f"runs/{exp}", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    ds_tr = DIV2KPairDataset(cfg["data"]["lr_train"], cfg["data"]["hr_train"],
                            scale=cfg["scale"], patch_size=cfg["data"]["patch_size"], training=True)
    ds_va = DIV2KPairDataset(cfg["data"]["lr_val"], cfg["data"]["hr_val"],
                            scale=cfg["scale"], patch_size=cfg["data"]["patch_size"], training=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg["data"]["batch_size"], shuffle=True,
                       num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    mcfg = cfg["model"]
    model = ViTSR158b(
        img_channels=mcfg.get("img_channels", 3),
        embed_dim=mcfg.get("embed_dim", 192),
        depth=mcfg.get("depth", 12),
        num_heads=mcfg.get("num_heads", 6),
        mlp_ratio=mcfg.get("mlp_ratio", 4.0),
        scale=cfg["scale"],
        act_bits=mcfg.get("quant", {}).get("act_bits", 8),
        drop=0.0,
        attn_drop=0.0,
    ).to(device)

    loss_fn = nn.L1Loss() if cfg.get("loss", {}).get("name", "l1") == "l1" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))

    amp_enabled = bool(cfg["train"].get("amp", True)) and device == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best = -1e9
    epochs = int(cfg["train"]["epochs"])
    val_every = int(cfg["train"].get("val_every", 1))
    save_every = int(cfg["train"].get("save_every", 10))

    fp_e = int(cfg["train"].get("ramp_fp_epochs", 3))
    act_e = int(cfg["train"].get("ramp_act_epochs", 3))

    set_model_quant_mode(model, act=False, w=False)
    print(f"[ramp] epochs 1..{fp_e}: FP (no quant)")
    print(f"[ramp] epochs {fp_e+1}..{fp_e+act_e}: act quant only")
    print(f"[ramp] epochs {fp_e+act_e+1}..: act + weight quant")

    for epoch in range(1, epochs + 1):
        fp_e  = int(cfg["train"].get("ramp_fp_epochs", 3))
        act_e = int(cfg["train"].get("ramp_act_epochs", 3))
        w_e   = int(cfg["train"].get("ramp_w_epochs", 10))

        if epoch <= fp_e:
            set_model_quant_mode(model, act=False, w=False, alpha_w=0.0)

        elif epoch <= fp_e + act_e:
            set_model_quant_mode(model, act=True, w=False, alpha_w=0.0)

        else:
            # pesos ternarios con alpha progresivo
            t = epoch - (fp_e + act_e)            # t = 1,2,3...
            alpha = min(1.0, t / max(1, w_e))     # 0->1 en w_e epochs
            set_model_quant_mode(model, act=True, w=True, alpha_w=alpha)

        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {epoch}/{epochs}")
        running = 0.0

        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                sr = model(lr)
                loss = loss_fn(sr, hr)

            if not torch.isfinite(loss):
                print(f"[nan] loss is not finite at epoch={epoch}. act={epoch>fp_e} w={epoch>fp_e+act_e}") # Por si me falla el loss
                return

            scaler.scale(loss).backward()

            grad_clip = float(cfg["train"].get("grad_clip", 0.0))
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=opt.param_groups[0]["lr"])

        if epoch % val_every == 0:
            tile = int(cfg.get("val", {}).get("tile", 64))
            overlap = int(cfg.get("val", {}).get("overlap", 16))
            max_images = cfg.get("val", {}).get("max_images", None)
            val_psnr = validate(model, dl_va, device, scale=cfg["scale"], tile=tile, overlap=overlap, max_images=max_images)
            print(f"[epoch {epoch}] val_psnr={val_psnr:.3f}")

            if val_psnr > best:
                best = val_psnr
                torch.save({"model": model.state_dict(), "cfg": cfg}, f"checkpoints/{exp}_best.pt")

        if epoch % save_every == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg}, f"checkpoints/{exp}_e{epoch}.pt")

    torch.save({"model": model.state_dict(), "cfg": cfg}, f"checkpoints/{exp}_last.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)