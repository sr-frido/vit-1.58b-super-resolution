# ViT-1.58b Super Resolution (x4)

Transformer-based super-resolution model with progressive quantization ramp:
- FP warmup
    then
- 8-bit activation quant
    then
- Ternary weight quantization

## Features
- Residual SR architecture
- Tile inference
- Quantization ramp
- Full-bit training

## Usage

Train:
```bash
python train.py --cfg configs/div2k_x4_vit158b.yaml

Run:

python app_sr.py --ckpt checkpoints/div2k_x4_vit158b_best.pt --web


Pendiente:
Descenso paulatino de la tasa de aprendizaje empezar en 1e-4 y terminar en 1e-5 por ejemplo?
Más bloques y menos tokens
Atención local en vez de global (aplicar ventantas tipo Swin)
¿Entrenar con ImageNet?
Entrenar con algo que no sea FP16 (arquitectura de la GPU)


Opcionales:
appWeb