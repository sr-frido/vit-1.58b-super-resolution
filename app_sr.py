import argparse
import os
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
import numpy as np

from src.models.vit_sr_158b import ViTSR158b


def load_model(ckpt_path: str, device: str = "cuda"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", None)

    # Intenta leer hiperparámetros del cfg si existen; si no, usa defaults razonables
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

    # Carga pesos
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    model.eval().to(device)

    # Si tu modelo usa BitLinear con set_quant, fuerza modo "full bit" para inferencia
    for m in model.modules():
        if hasattr(m, "set_quant"):
            # act=True, w=True, alpha_w=1.0 (si existe)
            try:
                m.set_quant(act=True, w=True, alpha_w=1.0)
            except TypeError:
                try:
                    m.set_quant(act=True, w=True)
                except TypeError:
                    pass

    return model


@torch.no_grad()
def tile_inference(model, lr, scale=4, tile=96, overlap=8, device="cuda"):
    """
    lr: (1,3,H,W) float [0,1]
    tile/overlap en píxeles LR
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
            sr_tile = model(lr_tile).clamp(0, 1)

            oy0, oy1 = y0 * scale, y1 * scale
            ox0, ox1 = x0 * scale, x1 * scale

            sr_acc[:, :, oy0:oy1, ox0:ox1] += sr_tile.float()
            wt_acc[:, :, oy0:oy1, ox0:ox1] += 1.0

            x += stride
        y += stride

    return (sr_acc / wt_acc.clamp_min(1e-8)).clamp(0, 1)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # HWC RGB [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # HWC
    arr = (t * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def run_sr(
    model,
    img_in: Image.Image,
    device="cuda",
    tile=96,
    overlap=8,
):
    lr = pil_to_tensor(img_in).to(device)

    # Lee scale desde el modelo
    scale = getattr(model, "scale", 4)

    # Inferencia por tiles (segura para imágenes grandes)
    sr = tile_inference(model, lr, scale=scale, tile=tile, overlap=overlap, device=device)
    out = tensor_to_pil(sr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Ruta al checkpoint .pt (por ej: checkpoints/div2k_x4_vit158b_best.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--overlap", type=int, default=8)

    # modo CLI
    ap.add_argument("--in", dest="inp", default=None, help="Imagen de entrada (modo CLI)")
    ap.add_argument("--out", dest="out", default=None, help="Ruta de salida (modo CLI)")

    # modo web
    ap.add_argument("--web", action="store_true", help="Lanza interfaz web con Gradio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)

    args = ap.parse_args()

    model = load_model(args.ckpt, device=args.device)

    # CLI
    if args.inp is not None:
        if args.out is None:
            raise SystemExit("En modo CLI necesitas --out salida.png")
        img = Image.open(args.inp)
        out = run_sr(model, img, device=args.device, tile=args.tile, overlap=args.overlap)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out.save(args.out)
        print("Saved:", args.out)
        return

    # Web
    if args.web:
        def gr_fn(img, tile, overlap):
            return run_sr(model, img, device=args.device, tile=int(tile), overlap=int(overlap))

        demo = gr.Interface(
            fn=gr_fn,
            inputs=[
                gr.Image(type="pil", label="Imagen LR (entrada)"),
                gr.Slider(32, 256, value=args.tile, step=8, label="Tile (LR px)"),
                gr.Slider(0, 64, value=args.overlap, step=4, label="Overlap (LR px)"),
            ],
            outputs=gr.Image(type="pil", label="SR (salida)"),
            title="Super-Resolución con ViT-1.58b (tile inference)",
            description="Sube una imagen y genera SR. Ajusta tile/overlap si va lento o da OOM.",
            flagging_mode="never",
        )
        demo.launch(server_name=args.host, server_port=args.port)
        return

    raise SystemExit("Usa --web para interfaz o --in/--out para CLI.")


if __name__ == "__main__":
    main()