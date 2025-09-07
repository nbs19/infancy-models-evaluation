# Saliency Map Evaluation for ResNet50 Models

This repository contains Python scripts for evaluating and comparing saliency maps between ResNet50 models and human attention maps on ImageNet datasets.


# Python Scripts for Data Processing

This repository contains Python scripts for processing click data and generating saliency maps for ImageNet datasets.

## Scripts Overview

### 1. combine_npz.py

Combines two `.npz` files into a single `.npz` file, merging arrays like `file_pointer`, `clickmap_x`, `clickmap_y`, and `user_id` while handling duplicates and ensuring consistency. Uses default output path if not specified.

**Arguments:**
- `--input1`: Path to the first input `.npz` file (required)
- `--input2`: Path to the second input `.npz` file (required)
- `--output`: Path to the output combined `.npz` file (default: `combined_click.npz` in current directory)
- `--verbose`: Enable verbose output to print processing details

**Example Usage:**
```bash
python3 combine_npz.py \
    --input1 /data/data0/Bhargava/prj_clickmev2_val_imagenet_10_10_2024.npz \
    --input2 /data/data0/Bhargava/prj_clickmev2_train_imagenet_10_10_2024.npz \
    --verbose
```

### 2. filter_npz.py

Filters a `.npz` file based on images present in an ImageNet validation directory, creating a new `.npz` file with matching data.

**Arguments:**
- `--input-npz`: Path to the input `.npz` dataset (required)
- `--imagenet-val-dir`: Path to the ImageNet validation directory containing images (required)
- `--output`: Path to the output filtered `.npz` file (required)

**Example Usage:**
```bash
python3 filter_npz.py \
    --input-npz /data/data0/Bhargava/combined_click.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \
    --output /data/data0/Bhargava/custom_filtered.npz
```

### 3. generate_saliency_maps.py

Generates saliency maps from click data in a `.npz` file using ImageNet validation images, with default output paths for the generated maps.

**Arguments:**
- `--input-npz`: Path to the input `.npz` dataset (required)
- `--imagenet-val-dir`: Path to the ImageNet validation directory containing images (required)
- `--output-dir`: Output directory for saliency maps (default: current directory)
- `--verbose`: Enable verbose output to print processing details

**Example Usage:**
```bash
python3 generate_saliency_maps.py \
    --input-npz /data/data0/Bhargava/filtered_dataset_1k.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \
    --verbose
```

## Output Files

The scripts generate:
- Combined `.npz` files with merged click data
- Filtered `.npz` files containing only matching ImageNet images
- Saliency map files generated from click data





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



### 4. inter_rater.py

Description: Calculates inter-rater alignment using split-half correlation on saliency maps derived from click data in a .npz file, filtering images with sufficient user inputs and processing ImageNet validation images. Runs in the background using nohup.
Arguments:
**Arguments:**
--input-npz: Path to the input .npz dataset (e.g., filtered_dataset_1k.npz) (required).
--imagenet-val-dir: Path to the ImageNet validation directory containing images (required).
--min-users: Minimum number of users required per image (default: 5).
--n-splits: Number of split-half iterations per image (default: 100).
--verbose: Enable verbose output to print processing details.

**Example Usage:**
 python3 inter_rater.py \
    --input-npz /home/bhargava/final_codes/human_align/filtered_dataset_1k.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \ 
    --min-users 3 \ 
    --n-splits 100 


