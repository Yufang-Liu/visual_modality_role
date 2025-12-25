# visual_modality_role

Code for the Paper "[The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](https://arxiv.org/pdf/2503.04167)". Data can be found in [here](https://drive.google.com/file/d/1jlR-0ZjAS0PbES8nz78BljI_lPMkEEfS/view?usp=sharing).

## Train

We examine recently proposed mathematical LVLMs, including G-LLaVA, MathLLaVA, MAVIS, and MultiMath. The corresponding execution scripts can be found in the `scripts` folder. For instance, to run G-LLaVA, simply execute:
```
sh gllava.sh
```
To modify the dataset used, simply adjust `--dataset_use` to the corresponding dataset in `llava/config/`. Image-shuffled or image-deleted versions of `geo170k` (training data of G-LLaVA) can be found in [here](https://huggingface.co/datasets/yfliu/shuffled_gllava). When employing multiple image encoders, modify `--vision_tower` to specify their combination:

- **Dot-separated** names (e.g., siglip.dino) indicate concatenation of hidden-layer representations.

- **Comma-separated** names (e.g., siglip,dino) denote concatenation of image tokens.

Examples:

- `scripts/gllava_siglip_dino.sh` (hidden-layer fusion)
- `scripts/gllava_siglip_dino_cat.sh` (token-level fusion)

