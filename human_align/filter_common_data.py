import os
import numpy as np
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments for input paths and other parameters."""
    parser = argparse.ArgumentParser(description="Filter ClickMe dataset based on images in ImageNet-10 validation directory.")
    parser.add_argument(
        "--input-npz",
        type=str,
        required=True,
        help="Path to the input .npz dataset (e.g., combined_click.npz)"
    )
    parser.add_argument(
        "--imagenet-val-dir",
        type=str,
        required=True,
        help="Path to the ImageNet-10 validation directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="filtered_dataset_1k.npz",
        help="Path to save the filtered .npz dataset (default: filtered_dataset_1k.npz)"
    )
    parser.add_argument(
        "--image-extension",
        type=str,
        default=".JPEG",
        help="File extension for images to include (default: .JPEG)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output to print dataset and processing details"
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Convert paths to Path objects for robust handling
    input_npz_path = Path(args.input_npz)
    imagenet_val_dir = Path(args.imagenet_val_dir)
    output_path = Path(args.output)

    # Validate input paths
    if not input_npz_path.is_file():
        raise FileNotFoundError(f"Input .npz file not found: {input_npz_path}")
    if not imagenet_val_dir.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {imagenet_val_dir}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the combined ClickMe dataset
    if args.verbose:
        print(f"Loading dataset from: {input_npz_path}")
    combined_data = np.load(input_npz_path, allow_pickle=True)

    # List all image names in the ImageNet-10 validation directory
    val_images = []
    for subdir in os.listdir(imagenet_val_dir):
        subdir_path = imagenet_val_dir / subdir
        if subdir_path.is_dir():  # Check if subdirectory exists
            for image_file in subdir_path.iterdir():
                if image_file.suffix.lower() == args.image_extension.lower():
                    val_images.append(image_file.name)

    # Check if any images were found
    if not val_images:
        raise ValueError(f"No images with extension {args.image_extension} found in {imagenet_val_dir}")
    
    if args.verbose:
        print(f"Found {len(val_images)} images in ImageNet validation directory")
        print(f"First few images in val_images: {val_images[:5]}")

    # Get the image names from the 'file_pointer' of the ClickMe data
    if 'file_pointer' not in combined_data:
        raise KeyError("'file_pointer' key not found in the input .npz file")
    
    file_pointers = combined_data['file_pointer']
    image_names = [os.path.basename(file_path) for file_path in file_pointers]

    # Find the common images between ClickMe data and ImageNet-10 val images
    valid_image_indices = [i for i, image_name in enumerate(image_names) if image_name in val_images]
    
    if not valid_image_indices:
        raise ValueError("No common images found between ClickMe dataset and ImageNet-10 validation images")
    
    if args.verbose:
        print(f"Found {len(valid_image_indices)} common images between datasets")

    # Create a new dictionary for the filtered data
    filtered_data = {key: combined_data[key][valid_image_indices] for key in combined_data.files}

    # Save the filtered data to a new .npz file
    np.savez(output_path, **filtered_data)
    print(f"Filtered dataset saved at: {output_path}")

    # Verify the saved data
    if args.verbose:
        saved_data = np.load(output_path, allow_pickle=True)
        print(f"Number of keys in the saved file: {len(saved_data.files)}")
        for key in saved_data:
            print(f"Key: {key}, Length of Array: {len(saved_data[key])}")

if __name__ == "__main__":
    main()