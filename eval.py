import argparse
import os
import random
import yaml
import torch
from torch.utils.data import DataLoader

from src.datasets.div2k import DIV2KPairDataset
from src.models.vit_sr_158b import ViTSR158b
from src.utils.metrics import psnr_rgb, psnr_y, ssim_rgb, ssim_y, LPIPSMetric


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None

    scale = 4
    embed_dim = 192
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    act_bits = 8

    if cfg is not None:
        scale = int(cfg.get("scale", scale))
        mcfg = cfg.get("model", {})
        embed_dim = int(mcfg.get("embed_dim", embed_dim))
        depth = int(mcfg.get("depth", depth))
        num_heads = int(mcfg.get("num_heads", num_heads))
        mlp_ratio = float(mcfg.get("mlp_ratio", mlp_ratio))
        act_bits = int(mcfg.get("quant", {}).get("act_bits", act_bits))

    model = ViTSR158b(
        img_channels=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        scale=scale,
        act_bits=act_bits,
        drop=0.0,
        attn_drop=0.0,
    )

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # Fuerza full-bit en inferencia si existe set_quant
    for m in model.modules():
        if hasattr(m, "set_quant"):
            try:
                m.set_quant(act=True, w=True, alpha_w=1.0)
            except TypeError:
                try:
                    m.set_quant(act=True, w=True)
                except TypeError:
                    pass

    return model, scale


@torch.inference_mode()
def tile_inference(model, lr, scale=4, tile=96, overlap=8, device="cuda"):
    """
    lr: [1,3,H,W] en [0,1]
    """
    model.eval()
    assert lr.dim() == 4 and lr.size(0) == 1, "tile_inference asume batch=1"
    _, _, H, W = lr.shape

    stride = tile - overlap
    if stride <= 0:
        raise ValueError("tile debe ser mayor que overlap")

    outH, outW = H * scale, W * scale
    sr_acc = torch.zeros((1, 3, outH, outW), device=device, dtype=torch.float32)
    wt_acc = torch.zeros((1, 1, outH, outW), device=device, dtype=torch.float32)

    y = 0
    while y < H:
        x = 0
        y0 = min(y, H - tile)
        y1 = y0 + tile
        while x < W:
            x0 = min(x, W - tile)
            x1 = x0 + tile

            lr_tile = lr[:, :, y0:y1, x0:x1]

            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    sr_tile = model(lr_tile)
            else:
                sr_tile = model(lr_tile)

            sr_tile = sr_tile.clamp(0, 1).float()

            oy0, oy1 = y0 * scale, y1 * scale
            ox0, ox1 = x0 * scale, x1 * scale

            sr_acc[:, :, oy0:oy1, ox0:ox1] += sr_tile
            wt_acc[:, :, oy0:oy1, ox0:ox1] += 1.0

            x += stride
        y += stride

    sr = sr_acc / wt_acc.clamp_min(1e-8)
    return sr.clamp(0, 1)


def make_val_loader(cfg, num_workers=2):
    ds = DIV2KPairDataset(
        cfg["data"]["lr_val"],
        cfg["data"]["hr_val"],
        scale=cfg["scale"],
        patch_size=cfg["data"]["patch_size"],
        training=False,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds, dl


def evaluate(
    model,
    dl,
    device="cuda",
    scale=4,
    tile=96,
    overlap=8,
    max_images=None,
    use_lpips=False,
    lpips_net="alex",
    seed=123,
):
    model.eval()

    # Si queremos una muestra aleatoria fija
    all_batches = list(dl)
    if max_images is not None and max_images < len(all_batches):
        rng = random.Random(seed)
        idxs = sorted(rng.sample(range(len(all_batches)), max_images))
        selected = [all_batches[i] for i in idxs]
    else:
        selected = all_batches

    lpips_metric = None
    if use_lpips:
        lpips_metric = LPIPSMetric(net=lpips_net, device=device)

    vals = {
        "psnr_rgb": [],
        "psnr_y": [],
        "ssim_rgb": [],
        "ssim_y": [],
        "lpips": [],
    }

    for i, (lr, hr) in enumerate(selected, start=1):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = tile_inference(model, lr, scale=scale, tile=tile, overlap=overlap, device=device)

        vals["psnr_rgb"].append(psnr_rgb(sr, hr).mean().item())
        vals["psnr_y"].append(psnr_y(sr, hr, shave=scale).mean().item())
        vals["ssim_rgb"].append(ssim_rgb(sr, hr).mean().item())
        vals["ssim_y"].append(ssim_y(sr, hr, shave=scale).mean().item())

        if lpips_metric is not None:
            vals["lpips"].append(lpips_metric(sr, hr).mean().item())

        print(
            f"[{i}/{len(selected)}] "
            f"PSNR_RGB={vals['psnr_rgb'][-1]:.3f} | "
            f"PSNR_Y={vals['psnr_y'][-1]:.3f} | "
            f"SSIM_RGB={vals['ssim_rgb'][-1]:.4f} | "
            f"SSIM_Y={vals['ssim_y'][-1]:.4f}"
            + (f" | LPIPS={vals['lpips'][-1]:.4f}" if lpips_metric is not None else "")
        )

    out = {
        "num_images": len(selected),
        "psnr_rgb": sum(vals["psnr_rgb"]) / max(1, len(vals["psnr_rgb"])),
        "psnr_y": sum(vals["psnr_y"]) / max(1, len(vals["psnr_y"])),
        "ssim_rgb": sum(vals["ssim_rgb"]) / max(1, len(vals["ssim_rgb"])),
        "ssim_y": sum(vals["ssim_y"]) / max(1, len(vals["ssim_y"])),
    }

    if lpips_metric is not None:
        out["lpips"] = sum(vals["lpips"]) / max(1, len(vals["lpips"]))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Ruta al YAML de config")
    ap.add_argument("--ckpt", required=True, help="Ruta al checkpoint .pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=10, help="Número de imágenes de validación a evaluar")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lpips", action="store_true", help="Activar LPIPS")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    model, scale = load_model(args.ckpt, device=args.device)
    _, dl = make_val_loader(cfg, num_workers=cfg["data"].get("num_workers", 2))

    results = evaluate(
        model=model,
        dl=dl,
        device=args.device,
        scale=scale,
        tile=args.tile,
        overlap=args.overlap,
        max_images=args.max_images,
        use_lpips=args.lpips,
        lpips_net=args.lpips_net,
        seed=args.seed,
    )

    print("\n===== RESULTADOS FINALES =====")
    print(f"Imágenes evaluadas: {results['num_images']}")
    print(f"PSNR_RGB : {results['psnr_rgb']:.3f} dB")
    print(f"PSNR_Y   : {results['psnr_y']:.3f} dB")
    print(f"SSIM_RGB : {results['ssim_rgb']:.4f}")
    print(f"SSIM_Y   : {results['ssim_y']:.4f}")
    if "lpips" in results:
        print(f"LPIPS    : {results['lpips']:.4f}")


if __name__ == "__main__":
    main()


#python eval.py \
#  --cfg configs/div2k_x4_vit158b.yaml \
#  --ckpt checkpoints/div2k_x4_vit158b_best.pt \
#  --max_images 50 \
#  --tile 96 \
#  --overlap 8 \
#  --lpips    usar alexnet para