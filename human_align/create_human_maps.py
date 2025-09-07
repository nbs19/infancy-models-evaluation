import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from PIL import Image
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(description="Generate saliency maps from ClickMe dataset and ImageNet-10 validation images.")
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
        help="Path to the ImageNet-10 validation directory containing images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="click_maps_emg",
        help="Directory to save saliency maps and user info (default: click_maps_emg)"
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default="click_maps_final_avg_1k.npz",
        help="Path to save the combined saliency maps .npz file (default: click_maps_final_avg_1k.npz)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output to print processing details"
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Define hardcoded parameters
    target_width, target_height = 224, 224
    sigma = 10.0
    image_extension = ".JPEG"

    # Convert paths to Path objects for robust handling
    input_npz_path = Path(args.input_npz)
    imagenet_val_dir = Path(args.imagenet_val_dir)
    output_dir = Path(args.output_dir)
    output_npz_path = Path(args.output_npz)

    # Validate input paths
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input .npz file not found: {input_npz_path}")
    if not imagenet_val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure output .npz path is absolute
    if not output_npz_path.is_absolute():
        output_npz_path = output_dir / output_npz_path

    # Load the filtered ClickMe dataset
    if args.verbose:
        print(f"Loading dataset from: {input_npz_path}")
    data = np.load(input_npz_path, allow_pickle=True)

    # Validate required keys
    required_keys = ['file_pointer', 'clickmap_x', 'clickmap_y', 'user_id']
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Missing required keys in .npz file: {missing_keys}")

    # Extract arrays
    file_pointers = data['file_pointer']
    clickmap_x = data['clickmap_x']
    clickmap_y = data['clickmap_y']
    user_ids = data['user_id']

    if args.verbose:
        print(f"Loaded {len(file_pointers)} entries from the dataset")

    # Create a dictionary to store the saliency map coordinates for each image
    image_user_map = defaultdict(list)

    # Populate the dictionary with (x, y) coordinates for each image
    for img, x_coords, y_coords, user in zip(file_pointers, clickmap_x, clickmap_y, user_ids):
        image_user_map[img].append((x_coords, y_coords))

    # Create a text file to store image name and number of users
    user_info_file = output_dir / 'image_user_info_1k.txt'
    with open(user_info_file, 'w') as f:
        f.write("Image Name\tNumber of Users\n")

    # Create a dictionary to store the normalized saliency maps
    normalized_saliency_maps = {}

    # Loop through each image to calculate and save the average saliency map
    for img, coords_list in image_user_map.items():
        # Extract the image name from file_pointer
        image_name = os.path.basename(img)

        # Find the image in the ImageNet val directory
        image_path = None
        for subdir in imagenet_val_dir.iterdir():
            candidate_path = subdir / image_name
            if candidate_path.is_file() and candidate_path.suffix.lower() == image_extension.lower():
                image_path = candidate_path
                break

        if image_path is None:
            if args.verbose:
                print(f"Warning: Image {image_name} not found in ImageNet val directory. Skipping.")
            continue

        # Get original image dimensions
        try:
            original_image = Image.open(image_path)
            original_width, original_height = original_image.size
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to open image {image_name}: {e}. Skipping.")
            continue

        # Create an empty saliency map
        saliency_map = np.zeros((target_height, target_width))

        # Accumulate the coordinates for each user
        for x_coords, y_coords in coords_list:
            # Scale coordinates to the target image size
            scaled_x = np.array(x_coords) * target_width / original_width
            scaled_y = np.array(y_coords) * target_height / original_height

            # Add the scaled coordinates to the saliency map
            for x, y in zip(scaled_x.astype(int), scaled_y.astype(int)):
                if 0 <= x < target_width and 0 <= y < target_height:
                    saliency_map[y, x] += 1

        # Normalize by number of users
        saliency_map = saliency_map / len(coords_list) if len(coords_list) > 0 else saliency_map

        # Apply Gaussian filter to smooth the map
        saliency_map = gaussian_filter(saliency_map, sigma=sigma)

        # Store the saliency map
        normalized_saliency_maps[image_name] = saliency_map

        # Write image name and number of users to the text file
        num_users = len(coords_list)
        with open(user_info_file, 'a') as f:
            f.write(f"{image_name}\t{num_users}\n")

        if args.verbose:
            print(f"Processed saliency map for {image_name} with {num_users} users")

    # Save all the normalized saliency maps to a single .npz file
    np.savez(output_npz_path, **normalized_saliency_maps)

    print(f"Saliency maps and user info saved in {output_dir} and {user_info_file}")
    if args.verbose:
        print(f"Combined saliency maps saved at: {output_npz_path}")
        print(f"Processed {len(normalized_saliency_maps)} images")

if __name__ == "__main__":
    main()