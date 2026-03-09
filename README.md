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
(Se abrirá en http://127.0.0.1:7860/)

Eval:

#python eval.py \
#  --cfg configs/div2k_x4_vit158b.yaml \    configuración del modelo 
#  --ckpt checkpoints/div2k_x4_vit158b_best.pt \ modelo a testear
#  --max_images 50 \    
#  --tile 96 \   tamaño del patch
#  --overlap 8 \  overlap entre patches
#  --lpips       usar alexnet como metrica de distancia perceptual <0.2 bueno  >0.5 kk


## Pendiente:
Corregir la evaluación de psnr en full RGB a solo luminancia
Añadir metricas de SSIM (Hay quien usa MSE pero no veo que de valores muy representativos de la calidad del modelo)
Descenso paulatino de la tasa de aprendizaje empezar en 1e-4 y terminar en 1e-5 por ejemplo?
Más bloques y menos tokens
Atención local en vez de global (aplicar ventantas tipo Swin)
¿Entrenar con ImageNet?
Entrenar con algo que no sea FP16 (arquitectura de la GPU) idealmente int8 y aprovechar la implementación de + en vez de *
¿Como hago que se de cuenta de perfilar más finas las lineas? (Quizá con la mejora de meter más bloques y meter ventana dentro del patch)

## Opcionales:
appWeb (primera versión lista)
