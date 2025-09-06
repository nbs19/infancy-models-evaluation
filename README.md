# infancy-models-evaluation
Python Scripts for Saliency Map Evaluation



:rocket: plot_two_model_img10.py



Description: Evaluates saliency maps for two ResNet50 models (current and baseline) on ImageNet-10 (10 classes), computes Spearman correlations with human saliency maps, and generates heatmap visualizations for all images with corresponding human saliency maps.


Arguments:  - --input-npz: Path to human saliency maps .npz file (required).  - --imagenet-val-dir: Path to ImageNet-10 validation directory (required).  - --current-model-checkpoint: Path to current model checkpoint (required).  - --baseline-model-checkpoint: Path to baseline model checkpoint (required).  - --output-dir: Output directory for heatmaps and correlations (default: results_check).  - --scores-file: Output file for correlations (default: correlation4_final.txt).  - --device-id: CUDA device ID (default: 0).  - --verbose: Enable detailed output.


Example Command:  bash <br> python3 plot_two_model_img10.py \ <br>     --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_10.npz \ # Human saliency maps file <br>     --imagenet-val-dir /data/data0/Bhargava/imagenet_10/val/ \ # ImageNet-10 validation images directory <br>     --current-model-checkpoint /home/bhargava/human_alignment/baby-linprobe-best-v7.ckpt \ # Current model checkpoint <br>     --baseline-model-checkpoint /home/bhargava/data_prep/shuffle-even-linprobe-best-v1.ckpt \ # Baseline model checkpoint <br>     --output-dir /home/bhargava/final_codes/human_align/imagenet_10_plots \ # Directory for output heatmaps and correlations <br>     --scores-file correlation4_final.txt \ # File to save correlation scores <br>     --device-id 0 \ # CUDA device (0 for cuda:0) <br>     --verbose # Print detailed processing info <br> 





:rocket: plot_two_model_img1k.py



Description: Evaluates saliency maps for two ResNet50 models on ImageNet-1k (1000 classes), selecting images from a correlation file (top 200, near-zero 200, most negative 200), with options to process all images or only correct predictions.


Arguments:  - --input-npz: Path to human saliency maps .npz file (required).  - --imagenet-val-dir: Path to ImageNet-1k validation directory (required).  - --correlation-file: Path to correlation file with image paths and correlations (required).  - --class-mapping-file: Path to class mapping file (required).  - --current-model-checkpoint: Path to current model checkpoint (required).  - --baseline-model-checkpoint: Path to baseline model checkpoint (required).  - --output-dir: Output directory (default: results_single_use).  - --scores-file: Output file for correlations (default: correlation4_gt.txt).  - --mode: correct_only (only correct predictions) or all (all images) (default: correct_only).  - --verbose: Enable detailed output.


Example Command:  bash <br> python3 plot_two_model_img1k.py \ <br>     --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz \ # Human saliency maps file <br>     --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \ # ImageNet-1k validation images directory <br>     --correlation-file /home/bhargava/final_codes/human_align/output/correlation_baby_1k.txt \ # File with image paths and correlations <br>     --class-mapping-file /home/bhargava/human_alignment/categories.txt \ # File mapping class IDs to names <br>     --current-model-checkpoint /data/data0/Bhargava/1k_models/baby_best.ckpt \ # Current model checkpoint <br>     --baseline-model-checkpoint /data/data0/Bhargava/1k_models/shuffle_best.ckpt \ # Baseline model checkpoint <br>     --output-dir /home/bhargava/final_codes/human_align/imagenet_1k_plots \ # Directory for output heatmaps and correlations <br>     --scores-file correlation4_gt.txt \ # File to save correlation scores <br>     --mode all \ # Process all images (not just correct predictions) <br>     --verbose # Print detailed processing info <br> 





:rocket: evaluate_saliency_maps.py



Description: Evaluates saliency maps for two ResNet50 models on ImageNet-1k, selecting images from a correlation file, with configurable number of classes and modes for correlation calculation.


Arguments:  - --input-npz: Path to human saliency maps .npz file (required).  - --imagenet-val-dir: Path to ImageNet-1k validation directory (required).  - --correlation-file: Path to correlation file (required).  - --class-mapping-file: Path to class mapping file (required).  - --current-model-checkpoint: Path to current model checkpoint (required).  - --baseline-model-checkpoint: Path to baseline model checkpoint (required).  - --output-dir: Output directory (default: results_single_use).  - --scores-file: Output file for correlations (default: correlation4_gt.txt).  - --mode: correct_only or all (default: correct_only).  - --num-classes: Number of output classes (default: 1000).  - --verbose: Enable detailed output.


Example Command:  bash <br> python3 evaluate_saliency_maps.py \ <br>     --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz \ # Human saliency maps file <br>     --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \ # ImageNet-1k validation images directory <br>     --correlation-file /home/bhargava/final_codes/human_align/output/correlation_baby_1k.txt \ # File with image paths and correlations <br>     --class-mapping-file /home/bhargava/human_alignment/categories.txt \ # File mapping class IDs to names <br>     --current-model-checkpoint /data/data0/Bhargava/1k_models/baby_best.ckpt \ # Current model checkpoint <br>     --baseline-model-checkpoint /data/data0/Bhargava/1k_models/shuffle_best.ckpt \ # Baseline model checkpoint <br>     --output-dir /home/bhargava/final_codes/human_align/imagenet_1k_plots \ # Directory for output heatmaps and correlations <br>     --scores-file correlation4_gt.txt \ # File to save correlation scores <br>     --mode all \ # Process all images (not just correct predictions) <br>     --num-classes 1000 \ # Set model output to 1000 classes <br>     --verbose # Print detailed processing info <br> 




Example Command:python3 plot_two_model_img10.py \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_10.npz \ # Human saliency maps file
    --imagenet-val-dir /data/data0/Bhargava/imagenet_10/val/ \ # ImageNet-10 validation images directory
    --current-model-checkpoint /home/bhargava/human_alignment/baby-linprobe-best-v7.ckpt \ # Current model checkpoint
    --baseline-model-checkpoint /home/bhargava/data_prep/shuffle-even-linprobe-best-v1.ckpt \ # Baseline model checkpoint
    --output-dir /home/bhargava/final_codes/human_align/imagenet_10_plots \ # Directory for output heatmaps and correlations
    --scores-file correlation4_final.txt \ # File to save correlation scores
    --device-id 0 \ # CUDA device (0 for cuda:0)
    --verbose # Print detailed processing info



2. plot_two_model_img1k.py

Description: Evaluates saliency maps for two ResNet50 models on ImageNet-1k (1000 classes), selecting images from a correlation file (top 200, near-zero 200, most negative 200), with options to process all images or only correct predictions.
Arguments:
--input-npz: Path to human saliency maps .npz file (required).
--imagenet-val-dir: Path to ImageNet-1k validation directory (required).
--correlation-file: Path to correlation file with image paths and correlations (required).
--class-mapping-file: Path to class mapping file (required).
--current-model-checkpoint: Path to current model checkpoint (required).
--baseline-model-checkpoint: Path to baseline model checkpoint (required).
--output-dir: Output directory (default: results_single_use).
--scores-file: Output file for correlations (default: correlation4_gt.txt).
--mode: correct_only (only correct predictions) or all (all images) (default: correct_only).
--verbose: Enable detailed output.


Example Command:python3 plot_two_model_img1k.py \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz \ # Human saliency maps file
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \ # ImageNet-1k validation images directory
    --correlation-file /home/bhargava/final_codes/human_align/output/correlation_baby_1k.txt \ # File with image paths and correlations
    --class-mapping-file /home/bhargava/human_alignment/categories.txt \ # File mapping class IDs to names
    --current-model-checkpoint /data/data0/Bhargava/1k_models/baby_best.ckpt \ # Current model checkpoint
    --baseline-model-checkpoint /data/data0/Bhargava/1k_models/shuffle_best.ckpt \ # Baseline model checkpoint
    --output-dir /home/bhargava/final_codes/human_align/imagenet_1k_plots \ # Directory for output heatmaps and correlations
    --scores-file correlation4_gt.txt \ # File to save correlation scores
    --mode all \ # Process all images (not just correct predictions)
    --verbose # Print detailed processing info



3. evaluate_saliency_maps.py

Description: Evaluates saliency maps for two ResNet50 models on ImageNet-1k, selecting images from a correlation file, with configurable number of classes and modes for correlation calculation.
Arguments:
--input-npz: Path to human saliency maps .npz file (required).
--imagenet-val-dir: Path to ImageNet-1k validation directory (required).
--correlation-file: Path to correlation file (required).
--class-mapping-file: Path to class mapping file (required).
--current-model-checkpoint: Path to current model checkpoint (required).
--baseline-model-checkpoint: Path to baseline model checkpoint (required).
--output-dir: Output directory (default: results_single_use).
--scores-file: Output file for correlations (default: correlation4_gt.txt).
--mode: correct_only or all (default: correct_only).
--num-classes: Number of output classes (default: 1000).
--verbose: Enable detailed output.


Example Command:python3 evaluate_saliency_maps.py \
    --input-npz /home/bhargava/final_codes/human_align/human_maps_output/human_click_maps_1k.npz \ # Human saliency maps file
    --imagenet-val-dir /data/data0/Bhargava/imagenet/val/ \ # ImageNet-1k validation images directory
    --correlation-file /home/bhargava/final_codes/human_align/output/correlation_baby_1k.txt \ # File with image paths and correlations
    --class-mapping-file /home/bhargava/human_alignment/categories.txt \ # File mapping class IDs to names
    --current-model-checkpoint /data/data0/Bhargava/1k_models/baby_best.ckpt \ # Current model checkpoint
    --baseline-model-checkpoint /data/data0/Bhargava/1k_models/shuffle_best.ckpt \ # Baseline model checkpoint
    --output-dir /home/bhargava/final_codes/human_align/imagenet_1k_plots \ # Directory for output heatmaps and correlations
    --scores-file correlation4_gt.txt \ # File to save correlation scores
    --mode all \ # Process all images (not just correct predictions)
    --num-classes 1000 \ # Set model output to 1000 classes
    --verbose # Print detailed processing info


