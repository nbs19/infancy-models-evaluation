import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from PIL import Image
from scipy.stats import spearmanr
import random
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments for input paths and parameters."""
    parser = argparse.ArgumentParser(description="Calculate inter-rater alignment using split-half correlation on saliency maps from click data.")
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to the input .npz dataset (e.g., filtered_dataset_1k.npz)"
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        required=True,
        help="Path to the ImageNet validation directory containing images"
    )
    parser.add_argument(
        "--min-users",
        type=int,
        default=5,
        help="Minimum number of users required per image (default: 5)"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=100,
        help="Number of split-half iterations per image (default: 100)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output to print processing details"
    )
    return parser.parse_args()

def create_individual_saliency_map(x_coords, y_coords, original_width, original_height):
    """Create a saliency map for a single user's clicks"""
    target_width, target_height = 224, 224
    saliency_map = np.zeros((target_height, target_width))
    
    # Scale coordinates to the target image size
    scaled_x = np.array(x_coords) * target_width / original_width
    scaled_y = np.array(y_coords) * target_height / original_height
    
    # Add the scaled coordinates to the saliency map
    for x, y in zip(scaled_x.astype(int), scaled_y.astype(int)):
        if 0 <= x < target_width and 0 <= y < target_height:
            saliency_map[y, x] += 1
    
    # Apply Gaussian filter to smooth the map
    saliency_map = gaussian_filter(saliency_map, sigma=10)
    
    return saliency_map

def calculate_split_half_correlation(user_maps):
    """Calculate correlation between two random halves of users"""
    if len(user_maps) < 2:
        return np.nan
    
    # Randomly split users into two groups
    users = list(range(len(user_maps)))
    random.shuffle(users)
    
    half1_size = len(users) // 2
    half1_users = users[:half1_size]
    half2_users = users[half1_size:2*half1_size]  # Ensure equal sizes
    
    # Create average maps for each half
    half1_map = np.mean([user_maps[i] for i in half1_users], axis=0)
    half2_map = np.mean([user_maps[i] for i in half2_users], axis=0)
    
    # Calculate Spearman correlation between the two halves
    half1_flat = half1_map.flatten()
    half2_flat = half2_map.flatten()
    
    correlation, _ = spearmanr(half1_flat, half2_flat)
    return correlation

def calculate_inter_rater_alignment(image_user_map, imagenet_val_path, n_splits=1000, min_users=3, verbose=False):
    """Calculate inter-rater alignment score using split-half method"""
    
    correlations = []
    
    for img_idx, (img, coords_list) in enumerate(image_user_map.items()):
        # Skip images with too few users
        if len(coords_list) < min_users:
            if verbose:
                print(f"Skipping image {os.path.basename(img)}: Only {len(coords_list)} users (min required: {min_users})")
            continue
            
        # Extract the image name from file_pointer
        image_name = os.path.basename(img)
        
        # Find the image in the ImageNet val directory
        image_path = None
        for subdir in os.listdir(imagenet_val_path):
            candidate_path = os.path.join(imagenet_val_path, subdir, image_name)
            if os.path.exists(candidate_path):
                image_path = candidate_path
                break
        
        if image_path is None:
            if verbose:
                print(f"Skipping image {image_name}: Not found in ImageNet val directory")
            continue
            
        # Get original image dimensions
        original_image = Image.open(image_path)
        original_width, original_height = original_image.size
        
        # Create individual saliency maps for each user
        user_maps = []
        for x_coords, y_coords in coords_list:
            user_map = create_individual_saliency_map(
                x_coords, y_coords, original_width, original_height
            )
            user_maps.append(user_map)
        
        # Calculate split-half correlations for this image
        image_correlations = []
        for split in range(min(n_splits, 100)):  # Limit splits per image for efficiency
            corr = calculate_split_half_correlation(user_maps)
            if not np.isnan(corr):
                image_correlations.append(corr)
        
        if image_correlations:
            correlations.extend(image_correlations)
        
        if verbose and (img_idx + 1) % 10 == 0:
            print(f"Processed {img_idx + 1} images")
    
    # Calculate mean inter-rater alignment
    if correlations:
        inter_rater_alignment = np.mean(correlations)
        std_alignment = np.std(correlations)
        return inter_rater_alignment, std_alignment, len(correlations)
    else:
        return 0.0, 0.0, 0

def main():
    # Parse arguments
    args = parse_arguments()

    # Convert paths to Path objects
    input_npz_path = Path(args.input_npz)
    imagenet_val_dir = Path(args.imagenet_val_dir)

    # Validate input paths
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input .npz file not found: {input_npz_path}")
    if not imagenet_val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")

    # Load data
    if args.verbose:
        print(f"Loading dataset from: {input_npz_path}")
    data = np.load(input_npz_path, allow_pickle=True)
    file_pointers = data['file_pointer']
    clickmap_x = data['clickmap_x']
    clickmap_y = data['clickmap_y']
    user_ids = data['user_id']

    # Create the image-user mapping
    image_user_map = defaultdict(list)
    for img, x_coords, y_coords, user in zip(file_pointers, clickmap_x, clickmap_y, user_ids):
        image_user_map[img].append((x_coords, y_coords))

    if args.verbose:
        print(f"Total images with click data: {len(image_user_map)}")

    # Filter only images with sufficient users
    filtered_images = {img: coords for img, coords in image_user_map.items() 
                      if len(coords) >= args.min_users}
    if args.verbose:
        print(f"Images with >= {args.min_users} users: {len(filtered_images)}")

    # Calculate inter-rater alignment
    print("Calculating inter-rater alignment score...")
    inter_rater_score, std_score, n_correlations = calculate_inter_rater_alignment(
        filtered_images, args.imagenet_val_dir, n_splits=args.n_splits, min_users=args.min_users, verbose=args.verbose
    )

    print(f"\nInter-rater Alignment Results:")
    print(f"Mean correlation: {inter_rater_score:.4f}")
    print(f"Standard deviation: {std_score:.4f}")
    print(f"Number of correlation measurements: {n_correlations}")
    print(f"Paper reported: ? = 0.66")

if __name__ == "__main__":
    main()