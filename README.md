## Saliency Map Evaluation for ResNet50 Models

This repository contains Python scripts for evaluating and comparing saliency maps between ResNet50 models and human attention maps on ImageNet datasets.


### Python Scripts for Data Processing

This repository contains Python scripts for processing click data and generating saliency maps for ImageNet datasets.

#### 1. combine_npz.py

Combines two `.npz` files into a single `.npz` file, merging arrays like `file_pointer`, `clickmap_x`, `clickmap_y`, and `user_id` while handling duplicates and ensuring consistency. Uses default output path if not specified.

**Arguments:**
- `--input1`: Path to the first input `.npz` file (required)
- `--input2`: Path to the second input `.npz` file (required)
- `--output`: Path to the output combined `.npz` file (default: `combined_click.npz` in current directory)

**Example Usage:**
```bash
python3 combine_npz.py
    --input1 /data/data0/Bhargava/prj_clickmev2_val_imagenet_10_10_2024.npz
    --input2 /data/data0/Bhargava/prj_clickmev2_train_imagenet_10_10_2024.npz
```

#### 2. filter_common_data.py

Filters a `.npz` file based on images present in an ImageNet validation directory, creating a new `.npz` file with matching data.

**Arguments:**
- `--input-npz`: Path to the input `.npz` dataset (required)
- `--imagenet-val-dir`: Path to the ImageNet validation directory containing images (required)
- `--output`: Path to the output filtered `.npz` file (required)

**Example Usage:**
```bash
python filter_common_data.py 
    --input-npz /home/bhargava/final_codes/human_align/combined_click.npz
    --imagenet-val-dir /data/data0/Bhargava/imagenet_10/val/
    --output /home/bhargava/final_codes/human_align/filtered_dataset_1k.npz
```

#### 3. create_human_maps.py

Generates saliency maps from click data in a `.npz` file using ImageNet validation images, with default output paths for the generated maps.

**Arguments:**
- `--input-npz`: Path to the input `.npz` dataset (required)
- `--imagenet-val-dir`: Path to the ImageNet validation directory containing images (required)
- `--output-dir`: Output directory for human maps
- `--output-npz`: Output file name

**Example Usage:**

```bash
python3 create_human_maps.py 
    --input-npz /data/data0/Bhargava/filtered_dataset_1k.npz 
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/
    --output-dir /home/bhargava/final_codes/human_align/output/
    --output-npz human_click_maps.npz
```

#### Output Files

The following scripts generate:
- Combined `.npz` files with merged click data
- Filtered `.npz` files containing only matching ImageNet images
- Attention maps generated from click data, a `.npz` file


### Calculating and Visualizing Human alignement

#### 1. only_corr_value.py

Computes correlation values (Spearman correlation) between saliency maps generated from a model checkpoint and human click data, processing ImageNet-10 validation images.

**Arguments:**
- `--imagenet-val-dir`: Path to the ImageNet validation directory containing images (required)
- `--input-npz`: Path to the input `.npz` file containing human click maps (required)
- `--checkpoint-path`: Path to the model checkpoint file (required)
- `--num-classes`: Number of output classes for the model (required)
- `--output-dir`: Output directory for correlation results (required)
- `--scores-file`: Output file for correlation scores (required)

**Example Usage for ImageNet-10:**
```bash
python3 only_corr_value.py \
    --imagenet-val-dir /data/data0/Bhargava/imagenet_10/val/ \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_10.npz \
    --checkpoint-path /home/bhargava/human_alignment/baby-linprobe-best-v7.ckpt \
    --num-classes 10 \
    --output-dir /home/bhargava/final_codes/human_align/output/ \
    --scores-file correlation_baby_10.txt
```

**Example Usage for ImageNet-1k:**
```bash
python3 only_corr_value.py \
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz \
    --checkpoint-path /data/data0/Bhargava/1k_models/baby_best.ckpt \
    --num-classes 1000 \
    --output-dir /home/bhargava/final_codes/human_align/output/ \
    --scores-file correlation_baby_1k.txt
```


#### 2. plot_two_model_img10.py

Evaluates saliency maps for two ResNet50 models on ImageNet-10 (10 classes), computes Spearman correlations with human saliency maps, and generates heatmap visualizations for all images with corresponding human saliency maps.

**Arguments:**
- `--input-npz`: Path to human saliency maps .npz file (required)
- `--imagenet-val-dir`: Path to ImageNet-10 validation directory (required)
- `--current-model-checkpoint`: Path to current model checkpoint (required)
- `--baseline-model-checkpoint`: Path to baseline model checkpoint (required)
- `--output-dir`: Output directory for heatmaps and correlations
- `--scores-file`: Output file for correlations
- `--device-id`: CUDA device ID (default: 0)

**Example Usage:**
```bash
python3 plot_two_model_img10.py \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_10.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet_10/val/ \
    --current-model-checkpoint /home/bhargava/human_alignment/baby-linprobe-best-v7.ckpt \
    --baseline-model-checkpoint /home/bhargava/data_prep/shuffle-even-linprobe-best-v1.ckpt \
    --output-dir /home/bhargava/final_codes/human_align/imagenet_10_plots \
    --scores-file correlation_final.txt \
    --device-id 0 
```

#### 3. plot_two_model_img1k.py

Visualize saliency maps for two ResNet50 models on ImageNet-1k (1000 classes), with options of selecting images from a correlation file (top 200, near-zero 200, most negative 200), to process all images or only correct predictions.

**Arguments:**
- `--input-npz`: Path to human saliency maps .npz file (required)
- `--imagenet-val-dir`: Path to ImageNet-1k validation directory (required)
- `--correlation-file`: Path to correlation file with image paths and correlations (required)
- `--class-mapping-file`: Path to class mapping file (required)
- `--current-model-checkpoint`: Path to current model checkpoint (required)
- `--baseline-model-checkpoint`: Path to baseline model checkpoint (required)
- `--output-dir`: Output directory (default: `results_single_use`)
- `--scores-file`: Output file for correlations 
- `--mode`: `correct_only` (only correct predictions) or `all` (all images) (default: `correct_only`)

**Example Usage:**
```bash
python3 plot_two_model_img1k.py
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/
    --correlation-file /home/bhargava/final_codes/human_align/output/correlation_baby_1k.txt
    --class-mapping-file /home/bhargava/human_alignment/categories.txt
    --current-model-checkpoint /data/data0/Bhargava/1k_models/baby_best.ckpt
    --baseline-model-checkpoint /data/data0/Bhargava/1k_models/shuffle_best.ckpt
    --output-dir /home/bhargava/final_codes/human_align/imagenet_1k_plots
    --scores-file correlation4_gt.txt
    --mode all
```
Note: This code takes very long time to run, as it visualizes features for all Imagenet-1k val set.
To visualize selected images only, you can comment out the following section in the code accordingly:
```bash
#selected_image_paths = np.concatenate([top_200, near_0_200, most_negative_200])
selected_image_paths = image_paths
#Comment out hardcoded paths for manual use later
#selected_image_paths = ['/data/data0/Bhargava/imagenet/val/n02104365/ILSVRC2012_val_00006842.JPEG', '/data/data0/Bhargava/imagenet/val/n01669191/ILSVRC2012_val_00010145.JPEG']
```
This allows you to process only specific images instead of the entire validation set, significantly reducing runtime.


#### 4. inter_rater.py

Calculates  Human inter-rater alignment (Consistency among human maps) using split-half correlation on attention maps derived from click data in a .npz file, filtering images with sufficient user inputs and processing ImageNet validation images. 

**Arguments:**
- `--input-npz`: Path to the input .npz dataset (e.g., filtered_dataset_1k.npz) (required)
- `--imagenet-val-dir`: Path to the ImageNet validation directory containing images (required)
- `--min-users`: Minimum number of users required per image to consider for calculation
- `--n-splits`: Number of split-half iterations per image (default: 100)

**Example Usage:**
```bash
 python3 inter_rater.py \
    --input-npz /home/bhargava/final_codes/human_align/filtered_dataset_1k.npz \
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \ 
    --min-users 3 \ 
    --n-splits 100 
```

