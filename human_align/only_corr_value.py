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
    parser = argparse.ArgumentParser(description="Compute saliency maps and Spearman correlations for a model against human saliency maps.")
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        required=True,
        help="Path to the ImageNet validation directory containing images"
    )
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to the input .npz file containing human saliency maps (e.g., click_maps_final_avg_1k.npz)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_final_1k",
        help="Directory to save correlation scores (default: results_final_1k)"
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default="correlation_baby_logits2.txt",
        help="File name for saving correlation scores (default: correlation_baby_logits2.txt)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model's checkpoint file (e.g., baby_best.ckpt)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of output classes for the model (default: 1000)"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=1,
        help="CUDA device ID to use (default: 1)"
    )
    parser.add_argument(
        "--image-extension",
        type=str,
        default=".JPEG",
        help="File extension for images in ImageNet directory (default: .JPEG)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output to print processing details"
    )
    return parser.parse_args()

def our_model(checkpoint_path, num_classes, device):
    """Load the model from checkpoint with configurable number of classes."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = checkpoint['state_dict']
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, num_classes)  # Use configurable num_classes
    
    # Filter backbone weights
    backbone_state_dict = {
        k.replace('model.backbone.', ''): v 
        for k, v in checkpoint.items() 
        if k.startswith('model.backbone.')
    }
    
    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier weights (shape should match [num_classes, 2048])
    if checkpoint['classification_head.weight'].shape == (num_classes, 2048):
        model.fc.weight.data = checkpoint['classification_head.weight']
        model.fc.bias.data = checkpoint['classification_head.bias']
    else:
        raise ValueError(f"Classifier weight shape mismatch: expected ({num_classes}, 2048), got {checkpoint['classification_head.weight'].shape}")
    
    return model

def main():
    # Parse arguments
    args = parse_arguments()

    # Convert paths to Path objects
    imagenet_val_dir = Path(args.imagenet_val_dir)
    input_npz_path = Path(args.input_npz)
    output_dir = Path(args.output_dir)
    scores_file = output_dir / args.scores_file
    checkpoint_path = Path(args.checkpoint_path)

    # Validate input paths
    if not imagenet_val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input .npz file not found: {input_npz_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

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

    # Load the model
    if args.verbose:
        print(f"Loading model from: {checkpoint_path} with {args.num_classes} classes")
    model = our_model(checkpoint_path, args.num_classes, device)
    model = model.to(device)
    model.eval()

    # Load saliency map dictionary
    npz_data = np.load(input_npz_path, allow_pickle=True)
    saliency_map_dict = {os.path.basename(k): v for k, v in npz_data.items()}
    if args.verbose:
        print(f"Loaded {len(saliency_map_dict)} human saliency maps")

    # Initialize data collection
    spearman_scores = []
    correlation_data = []

    # Process images
    for subdir in os.listdir(imagenet_val_dir):
        subdir_path = Path(imagenet_val_dir) / subdir
        if subdir_path.is_dir():
            for image_file in os.listdir(subdir_path):
                if image_file.lower().endswith(args.image_extension.lower()):
                    image_name = image_file
                    if image_name not in saliency_map_dict:
                        if args.verbose:
                            print(f"Skipping {image_name}: Not found in saliency map dictionary")
                        continue

                    # Load original image
                    image_path = subdir_path / image_file
                    try:
                        original_image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(original_image).unsqueeze(0).to(device)
                    except Exception as e:
                        if args.verbose:
                            print(f"Skipping {image_name}: Failed to load image ({e})")
                        continue

                    image_tensor.requires_grad_(True)

                    # Forward pass with gradient tracking
                    with torch.enable_grad():
                        logits = model(image_tensor)
                        max_logit_idx = logits.argmax(dim=1)
                        target_logit = logits[0, max_logit_idx]
                        target_logit.backward()

                    # Get gradients and compute saliency map
                    gradients = image_tensor.grad.abs()
                    saliency_map = gradients.squeeze().mean(dim=0).cpu().numpy()
                    saliency_map = gaussian_filter(saliency_map, sigma=10)

                    # Load human saliency map
                    human_saliency_map_orig = saliency_map_dict[image_name].squeeze()

                    # Compute Spearman correlation
                    correlation, _ = spearmanr(saliency_map.flatten(), human_saliency_map_orig.flatten())

                    if np.isnan(correlation):
                        if args.verbose:
                            print(f"Warning: NaN correlation for image {image_name}")
                    else:
                        spearman_scores.append(correlation)

                    correlation_data.append(f"{image_path}\t{correlation:.6f}")

                    if args.verbose:
                        print(f"Processed {image_name}: Spearman Correlation = {correlation:.6f}")

    print("Processing complete")

    # Write correlation data to file
    with open(scores_file, 'w') as f:
        f.write("Image Name\tSpearman Correlation\n")
        f.write("\n".join(correlation_data))

    if spearman_scores:
        average_spearman = np.mean(spearman_scores)
        print(f"Average Spearman Correlation: {average_spearman:.6f}")
    else:
        print("No valid Spearman correlations computed")

    print(f"Correlation data written to: {scores_file}")

if __name__ == "__main__":
    main()