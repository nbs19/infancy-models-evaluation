import torch
from torch import nn
from torchvision.models import resnet50
from PIL import Image
import torchvision.models as models
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.stats import spearmanr
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)

def parse_arguments():
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(description="Evaluate saliency maps for current and baseline models against human saliency maps.")
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to the input .npz file containing human saliency maps (e.g., click_maps_final_avg_1k.npz)"
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        required=True,
        help="Path to the ImageNet-10 validation directory containing images"
    )
    parser.add_argument(
        "--correlation-file",
        type=str,
        required=True,
        help="Path to the correlation file with image paths and correlations"
    )
    parser.add_argument(
        "--class-mapping-file",
        type=str,
        required=True,
        help="Path to the class mapping text file (e.g., categories.txt)"
    )
    parser.add_argument(
        "--current-model-checkpoint",
        type=str,
        required=True,
        help="Path to the current model's checkpoint file (e.g., baby_best.ckpt)"
    )
    parser.add_argument(
        "--baseline-model-checkpoint",
        type=str,
        required=True,
        help="Path to the baseline model's checkpoint file (e.g., shuffle_best.ckpt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_single_use",
        help="Directory to save correlation scores and heatmap images (default: results_single_use)"
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default="correlation4_gt.txt",
        help="File name for saving correlation scores (default: correlation4_gt.txt)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="correct_only",
        choices=["correct_only", "all"],
        help="Mode for correlation calculation: 'correct_only' (only for correct predictions) or 'all' (for all images irrespective of ground truth label) (default: correct_only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output to print processing details"
    )
    return parser.parse_args()

def our_model(checkpoint_path, device):
    """Load the current model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = checkpoint['state_dict']
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 1000)  # 1000 classes
    
    # Filter backbone weights
    backbone_state_dict = {
        k.replace('model.backbone.', ''): v 
        for k, v in checkpoint.items() 
        if k.startswith('model.backbone.')
    }
    
    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier weights
    if checkpoint['classification_head.weight'].shape == (1000, 2048):
        model.fc.weight.data = checkpoint['classification_head.weight']
        model.fc.bias.data = checkpoint['classification_head.bias']
    
    return model

def baseline_model(checkpoint_path, device):
    """Load the baseline model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = checkpoint['state_dict']
    
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 1000)  # 1000 classes
    
    # Filter backbone weights
    backbone_state_dict = {
        k.replace('model.backbone.', ''): v 
        for k, v in checkpoint.items() 
        if k.startswith('model.backbone.')
    }
    
    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier weights
    if checkpoint['classification_head.weight'].shape == (1000, 2048):
        model.fc.weight.data = checkpoint['classification_head.weight']
        model.fc.bias.data = checkpoint['classification_head.bias']
    
    return model

def load_class_mapping(mapping_file_path):
    """Load class mapping from text file."""
    class_to_idx = {}
    idx_to_class = {}
    idx_to_name = {}
    
    with open(mapping_file_path, 'r') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_id = parts[0]
                class_name = parts[1]
                class_to_idx[class_id] = idx
                idx_to_class[idx] = class_id
                idx_to_name[idx] = class_name
    
    return class_to_idx, idx_to_class, idx_to_name

def get_ground_truth_label(image_path, class_to_idx):
    """Extract ground truth label from ImageNet validation file path structure."""
    class_folder = os.path.basename(os.path.dirname(image_path))
    return class_to_idx.get(class_folder, None)

def main():
    # Parse arguments
    args = parse_arguments()

    # Convert paths to Path objects
    input_npz_path = Path(args.input_npz)
    imagenet_val_dir = Path(args.imagenet_val_dir)
    correlation_file = Path(args.correlation_file)
    class_mapping_file = Path(args.class_mapping_file)
    current_model_checkpoint = Path(args.current_model_checkpoint)
    baseline_model_checkpoint = Path(args.baseline_model_checkpoint)
    output_dir = Path(args.output_dir)
    scores_file = output_dir / args.scores_file

    # Validate input paths
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input .npz file not found: {input_npz_path}")
    if not imagenet_val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")
    if not correlation_file.is_file():
        raise FileNotFoundError(f"Correlation file not found: {correlation_file}")
    if not class_mapping_file.is_file():
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_file}")
    if not current_model_checkpoint.is_file():
        raise FileNotFoundError(f"Current model checkpoint not found: {current_model_checkpoint}")
    if not baseline_model_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline model checkpoint not found: {baseline_model_checkpoint}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if args.verbose:
        print(f"Using device: {device}")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load models
    if args.verbose:
        print(f"Loading current model from: {current_model_checkpoint}")
    current_model = our_model(current_model_checkpoint, device)
    current_model = current_model.to(device)
    current_model.eval()

    if args.verbose:
        print(f"Loading baseline model from: {baseline_model_checkpoint}")
    baseline = baseline_model(baseline_model_checkpoint, device)
    baseline = baseline.to(device)
    baseline.eval()

    # Load class mapping
    class_to_idx, idx_to_class, idx_to_name = load_class_mapping(class_mapping_file)

    # Load saliency map dictionary
    npz_data = np.load(input_npz_path, allow_pickle=True)
    saliency_map_dict = {os.path.basename(k): v for k, v in npz_data.items()}
    if args.verbose:
        print(f"Loaded {len(saliency_map_dict)} human saliency maps")

    # Read correlation file
    image_paths = []
    correlations = []
    try:
        with open(correlation_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        image_paths.append(parts[0])
                        correlations.append(float(parts[1]))
                    except ValueError as e:
                        if args.verbose:
                            print(f"Error processing line: {line}, Error: {e}")
                        continue
    except Exception as e:
        raise RuntimeError(f"Failed to read correlation file: {e}")

    if not image_paths:
        raise ValueError("No valid image paths found in correlation file")

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    correlations = np.array(correlations)

    # Select top 200, near-zero 200, most negative 200
    top_200 = image_paths[correlations.argsort()[-200:]]
    near_0_200 = image_paths[np.argsort(np.abs(correlations))[:200]]
    most_negative_200 = image_paths[correlations.argsort()[:200]]

    #selected_image_paths = np.concatenate([top_200, near_0_200, most_negative_200])
    selected_image_paths = image_paths

    # Comment out hardcoded paths for manual use later
    # selected_image_paths = [
    #     '/data/data0/Bhargava/imagenet/val/n02104365/ILSVRC2012_val_00006842.JPEG',
    #     '/data/data0/Bhargava/imagenet/val/n01669191/ILSVRC2012_val_00010145.JPEG'
    # ]

    if args.verbose:
        print(f"Processing {len(selected_image_paths)} selected images")

    # Initialize data collection
    spearman_scores_current = []
    spearman_scores_baseline = []
    correlation_data = []
    correct_predictions_count = 0
    total_processed = 0

    for image_path in selected_image_paths:
        image_path = Path(image_path)
        if not image_path.is_file():
            if args.verbose:
                print(f"Skipping {image_path}: File not found")
            continue

        image_name = image_path.name
        if image_name not in saliency_map_dict:
            if args.verbose:
                print(f"Skipping {image_name}: Not found in saliency map dictionary")
            continue

        # Load and preprocess image
        try:
            original_image = Image.open(image_path).convert('RGB')
            image_tensor = transform(original_image).unsqueeze(0).to(device)
            image_tensor2 = transform(original_image).unsqueeze(0).to(device)
        except Exception as e:
            if args.verbose:
                print(f"Skipping {image_name}: Failed to load image ({e})")
            continue

        # Get ground truth label
        ground_truth_label = get_ground_truth_label(image_path, class_to_idx)
        if ground_truth_label is None:
            if args.verbose:
                print(f"Skipping {image_name}: Invalid ground truth label")
            continue

        # Get current model prediction
        with torch.no_grad():
            logits_current_check = current_model(image_tensor.clone())
            predicted_label = logits_current_check.argmax(dim=1).item()

        # Skip if mode is 'correct_only' and prediction doesn't match ground truth
        if args.mode == "correct_only" and predicted_label != ground_truth_label:
            if args.verbose:
                print(f"Skipping {image_name}: Incorrect prediction (predicted: {predicted_label}, ground truth: {ground_truth_label})")
            continue

        total_processed += 1
        if ground_truth_label is not None and predicted_label == ground_truth_label:
            correct_predictions_count += 1

        # Enable gradients
        image_tensor.requires_grad_(True)
        image_tensor2.requires_grad_(True)

        # Baseline model saliency map
        with torch.enable_grad():
            logits_baseline = baseline(image_tensor2)
            max_logit_idx_baseline = logits_baseline.argmax(dim=1)
            target_logit_baseline = logits_baseline[0, max_logit_idx_baseline]
            target_logit_baseline.backward()

        gradients_baseline = image_tensor2.grad.abs()
        saliency_map_baseline = gradients_baseline.squeeze().mean(dim=0).cpu().numpy()
        saliency_map_baseline = gaussian_filter(saliency_map_baseline, sigma=10)

        # Current model saliency map
        with torch.enable_grad():
            logits_current = current_model(image_tensor)
            max_logit_idx_current = logits_current.argmax(dim=1)
            target_logit_current = logits_current[0, max_logit_idx_current]
            target_logit_current.backward()

        gradients_current = image_tensor.grad.abs()
        saliency_map_current = gradients_current.squeeze().mean(dim=0).cpu().numpy()
        saliency_map_current = gaussian_filter(saliency_map_current, sigma=10)

        # Load human saliency map
        human_saliency_map_orig = saliency_map_dict[image_name].squeeze()

        # Compute Spearman correlation
        correlation_current, _ = spearmanr(saliency_map_current.flatten(), human_saliency_map_orig.flatten())
        correlation_baseline, _ = spearmanr(saliency_map_baseline.flatten(), human_saliency_map_orig.flatten())

        if np.isnan(correlation_current):
            if args.verbose:
                print(f"Warning: NaN correlation for image {image_name} (Current Model)")
        else:
            spearman_scores_current.append(correlation_current)
        if np.isnan(correlation_baseline):
            if args.verbose:
                print(f"Warning: NaN correlation for image {image_name} (Baseline Model)")
        else:
            spearman_scores_baseline.append(correlation_baseline)

        # Store correlation data
        correlation_data.append(f"{image_name}\t{correlation_current:.6f}\t{correlation_baseline:.6f}")

        # Normalize saliency maps for visualization
        saliency_map_vis_current = (saliency_map_current - saliency_map_current.min()) / (saliency_map_current.max() - saliency_map_current.min() + 1e-8)
        saliency_map_vis_baseline = (saliency_map_baseline - saliency_map_baseline.min()) / (saliency_map_baseline.max() - saliency_map_baseline.min() + 1e-8)
        human_saliency_vis = (human_saliency_map_orig - human_saliency_map_orig.min()) / (human_saliency_map_orig.max() - human_saliency_map_orig.min() + 1e-8)

        # Visualization
        plt.rcParams["font.family"] = "DejaVu Serif"
        plt.rcParams["font.size"] = 20
        fig, axes = plt.subplots(1, 4, figsize=(28, 7))

        # Resize original image
        original_image_224 = original_image.resize((224, 224))
        image_np = np.array(original_image_224)

        # Plot original image
        fs = 35
        axes[0].imshow(image_np)
        axes[0].set_title("Original Image", fontsize=fs)
        axes[0].axis("off")

        # Plot current model saliency
        axes[1].imshow(image_np)
        im2 = axes[1].imshow(saliency_map_vis_current, cmap="jet", alpha=0.5)
        axes[1].set_title(f"DEV\nSpearman: {correlation_current:.3f}", fontsize=fs)
        axes[1].axis("off")

        # Plot baseline model saliency
        axes[2].imshow(image_np)
        axes[2].imshow(saliency_map_vis_baseline, cmap="jet", alpha=0.5)
        axes[2].set_title(f"SHF\nSpearman: {correlation_baseline:.3f}", fontsize=fs)
        axes[2].axis("off")

        # Plot human saliency
        axes[3].imshow(image_np)
        axes[3].imshow(human_saliency_vis, cmap="jet", alpha=0.5)
        axes[3].set_title("Human Saliency Map", fontsize=fs)
        axes[3].axis("off")

        # Add colorbar
        cbar = fig.colorbar(im2, ax=axes[1:], fraction=0.046, pad=0.04)
        cbar.set_label("Saliency Intensity", rotation=270, labelpad=40, fontsize=fs)
        cbar.ax.tick_params(labelsize=18)

        # Save plot
        output_path = output_dir / f"heatmap_{image_name}.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=600)
        plt.close()

        if args.verbose:
            print(f"Processed {image_name}: Current={correlation_current:.6f}, Baseline={correlation_baseline:.6f}")

    # Write correlation data to file
    with open(scores_file, 'w') as f:
        f.write("Image Name\tCurrent Model Correlation\tBaseline Model Correlation\n")
        for line in correlation_data:
            f.write(f"{line}\n")

    # Compute and display results
    print("Processing complete")
    print(f"Total images processed: {total_processed}")
    print(f"Images with correct predictions: {correct_predictions_count}")

    if spearman_scores_current:
        average_spearman_current = np.mean(spearman_scores_current)
        print(f"Average Spearman Correlation (Current Model): {average_spearman_current:.6f}")
    else:
        print("No valid Spearman correlations for Current Model")
    if spearman_scores_baseline:
        average_spearman_baseline = np.mean(spearman_scores_baseline)
        print(f"Average Spearman Correlation (Baseline Model): {average_spearman_baseline:.6f}")
    else:
        print("No valid Spearman correlations for Baseline Model")

    print(f"Correlation data written to: {scores_file}")

if __name__ == "__main__":
    main()