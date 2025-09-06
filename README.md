# Saliency Map Evaluation for ResNet50 Models

This repository contains Python scripts for evaluating and comparing saliency maps between ResNet50 models and human attention maps on ImageNet datasets.

## Scripts Overview

### 1. plot_two_model_img10.py

Evaluates saliency maps for two ResNet50 models on ImageNet-10 (10 classes), computes Spearman correlations with human saliency maps, and generates heatmap visualizations for all images with corresponding human saliency maps.

**Arguments:**
- `--input-npz`: Path to human saliency maps .npz file (required)
- `--imagenet-val-dir`: Path to ImageNet-10 validation directory (required)
- `--current-model-checkpoint`: Path to current model checkpoint (required)
- `--baseline-model-checkpoint`: Path to baseline model checkpoint (required)
- `--output-dir`: Output directory for heatmaps and correlations (default: `results_check`)
- `--scores-file`: Output file for correlations (default: `correlation4_final.txt`)
- `--device-id`: CUDA device ID (default: 0)
- `--verbose`: Enable detailed output

**Example Usage:**
```bash
python3 plot_two_model_img10.py \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_10.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet_10/val/ \
    --current-model-checkpoint /home/bhargava/human_alignment/baby-linprobe-best-v7.ckpt \
    --baseline-model-checkpoint /home/bhargava/data_prep/shuffle-even-linprobe-best-v1.ckpt \
    --output-dir /home/bhargava/final_codes/human_align/imagenet_10_plots \
    --scores-file correlation4_final.txt \
    --device-id 0 \
    --verbose
```

### 2. plot_two_model_img1k.py

Evaluates saliency maps for two ResNet50 models on ImageNet-1k (1000 classes), selecting images from a correlation file (top 200, near-zero 200, most negative 200), with options to process all images or only correct predictions.

**Arguments:**
- `--input-npz`: Path to human saliency maps .npz file (required)
- `--imagenet-val-dir`: Path to ImageNet-1k validation directory (required)
- `--correlation-file`: Path to correlation file with image paths and correlations (required)
- `--class-mapping-file`: Path to class mapping file (required)
- `--current-model-checkpoint`: Path to current model checkpoint (required)
- `--baseline-model-checkpoint`: Path to baseline model checkpoint (required)
- `--output-dir`: Output directory (default: `results_single_use`)
- `--scores-file`: Output file for correlations (default: `correlation4_gt.txt`)
- `--mode`: `correct_only` (only correct predictions) or `all` (all images) (default: `correct_only`)
- `--verbose`: Enable detailed output

**Example Usage:**
```bash
python3 plot_two_model_img1k.py \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \
    --correlation-file /home/bhargava/final_codes/human_align/output/correlation_baby_1k.txt \
    --class-mapping-file /home/bhargava/human_alignment/categories.txt \
    --current-model-checkpoint /data/data0/Bhargava/1k_models/baby_best.ckpt \
    --baseline-model-checkpoint /data/data0/Bhargava/1k_models/shuffle_best.ckpt \
    --output-dir /home/bhargava/final_codes/human_align/imagenet_1k_plots \
    --scores-file correlation4_gt.txt \
    --mode all \
    --verbose
```

### 3. evaluate_saliency_maps.py

Evaluates saliency maps for two ResNet50 models on ImageNet-1k, selecting images from a correlation file, with configurable number of classes and modes for correlation calculation.

**Arguments:**
- `--input-npz`: Path to human saliency maps .npz file (required)
- `--imagenet-val-dir`: Path to ImageNet-1k validation directory (required)
- Additional configurable parameters for flexible evaluation

## Output Files

Each script generates:
- Heatmap visualizations comparing model saliency maps with human attention maps
- Correlation score files containing Spearman correlation coefficients
- Summary statistics and analysis results
