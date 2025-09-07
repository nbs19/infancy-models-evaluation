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
    parser = argparse.ArgumentParser(description="Compute saliency maps and Spearman correlations for current and baseline models against human saliency maps.")
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to the input .npz file containing human saliency maps (e.g., click_maps_no_norm.npz)"
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        required=True,
        help="Path to the ImageNet-10 validation directory containing images"
    )
    parser.add_argument(
        "--current-model-checkpoint",
        type=str,
        required=True,
        help="Path to the current model's checkpoint file (e.g., baby-linprobe-best-v7.ckpt)"
    )
    parser.add_argument(
        "--baseline-model-checkpoint",
        type=str,
        required=True,
        help="Path to the baseline model's checkpoint file (e.g., shuffle-even-linprobe-best-v1.ckpt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_check",
        help="Directory to save correlation scores and heatmap images (default: results_check)"
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default="correlation4_final.txt",
        help="File name for saving correlation scores (default: correlation4_final.txt)"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device ID to use (default: 0)"
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
    model.fc = torch.nn.Linear(2048, 10)  # Fixed to 10 classes
    
    # Filter backbone weights
    backbone_state_dict = {
        k.replace('model.backbone.', ''): v 
        for k, v in checkpoint.items() 
        if k.startswith('model.backbone.')
    }
    
    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier weights
    if checkpoint['classification_head.weight'].shape == (10, 2048):
        model.fc.weight.data = checkpoint['classification_head.weight']
        model.fc.bias.data = checkpoint['classification_head.bias']
    else:
        raise ValueError(f"Classifier weight shape mismatch: expected (10, 2048), got {checkpoint['classification_head.weight'].shape}")
    
    return model

def baseline_model(checkpoint_path, device):
    """Load the baseline model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = checkpoint['state_dict']
    
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 10)  # Fixed to 10 classes
    
    # Filter backbone weights
    backbone_state_dict = {
        k.replace('model.backbone.', ''): v 
        for k, v in checkpoint.items() 
        if k.startswith('model.backbone.')
    }
    
    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier weights
    if checkpoint['classification_head.weight'].shape == (10, 2048):
        model.fc.weight.data = checkpoint['classification_head.weight']
        model.fc.bias.data = checkpoint['classification_head.bias']
    else:
        raise ValueError(f"Classifier weight shape mismatch: expected (10, 2048), got {checkpoint['classification_head.weight'].shape}")
    
    return model

def main():
    # Parse arguments
    args = parse_arguments()

    # Convert paths to Path objects
    input_npz_path = Path(args.input_npz)
    imagenet_val_dir = Path(args.imagenet_val_dir)
    current_model_checkpoint = Path(args.current_model_checkpoint)
    baseline_model_checkpoint = Path(args.baseline_model_checkpoint)
    output_dir = Path(args.output_dir)
    scores_file = output_dir / args.scores_file

    # Validate input paths
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input .npz file not found: {input_npz_path}")
    if not imagenet_val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")
    if not current_model_checkpoint.is_file():
        raise FileNotFoundError(f"Current model checkpoint not found: {current_model_checkpoint}")
    if not baseline_model_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline model checkpoint not found: {baseline_model_checkpoint}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else 'cpu')
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

    # Load saliency map dictionary
    npz_data = np.load(input_npz_path, allow_pickle=True)
    saliency_map_dict = {os.path.basename(k): v for k, v in npz_data.items()}
    if args.verbose:
        print(f"Loaded {len(saliency_map_dict)} human saliency maps")

    # Initialize data collection
    spearman_scores_current = []
    spearman_scores_baseline = []
    correlation_data = []

    # Process images
    for subdir in os.listdir(imagenet_val_dir):
        subdir_path = Path(imagenet_val_dir) / subdir
        if subdir_path.is_dir():
            for image_file in os.listdir(subdir_path):
                if image_file.endswith('.JPEG'):
                    image_name = image_file
                    if image_name not in saliency_map_dict:
                        if args.verbose:
                            print(f"Skipping {image_name}: Not found in saliency map dictionary")
                        continue

                    # Load and preprocess image
                    image_path = subdir_path / image_file
                    try:
                        original_image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(original_image).unsqueeze(0).to(device)
                        image_tensor2 = transform(original_image).unsqueeze(0).to(device)
                    except Exception as e:
                        if args.verbose:
                            print(f"Skipping {image_name}: Failed to load image ({e})")
                        continue

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
                    plt.figure(figsize=(18, 6))
                    original_image_224 = original_image.resize((224, 224))
                    image_np = np.array(original_image_224)

                    # Plot original image
                    plt.subplot(1, 4, 1)
                    plt.imshow(image_np)
                    plt.title('Original Image')
                    plt.axis('off')

                    # Plot current model heatmap
                    plt.subplot(1, 4, 2)
                    plt.imshow(image_np)
                    im2 = plt.imshow(saliency_map_vis_current, cmap='jet', alpha=0.5)
                    plt.title(f'Current Model\nSpearman: {correlation_current:.3f}')
                    plt.axis('off')
                    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
                    cbar2.set_label('Saliency Intensity', rotation=270, labelpad=15)

                    # Plot baseline model heatmap
                    plt.subplot(1, 4, 3)
                    plt.imshow(image_np)
                    im2 = plt.imshow(saliency_map_vis_baseline, cmap='jet', alpha=0.5)
                    plt.title(f'Baseline Model\nSpearman: {correlation_baseline:.3f}')
                    plt.axis('off')
                    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
                    cbar2.set_label('Saliency Intensity', rotation=270, labelpad=15)

                    # Plot human saliency map
                    plt.subplot(1, 4, 4)
                    plt.imshow(image_np)
                    im2 = plt.imshow(human_saliency_vis, cmap='jet', alpha=0.5)
                    plt.title('Human Saliency Map')
                    plt.axis('off')
                    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
                    cbar2.set_label('Saliency Intensity', rotation=270, labelpad=15)

                    # Save visualization
                    output_path = output_dir / f'heatmap_{image_name}.png'
                    plt.savefig(output_path, bbox_inches='tight')
                    plt.close()

                    if args.verbose:
                        print(f"Processed {image_name}: Current={correlation_current:.6f}, Baseline={correlation_baseline:.6f}")

    # Write correlation data to file
    with open(scores_file, 'w') as f:
        f.write("Image Name\tCurrent Model Spearman Correlation\tBaseline Model Spearman Correlation\n")
        f.write("\n".join(correlation_data))

    print("Processing complete")
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