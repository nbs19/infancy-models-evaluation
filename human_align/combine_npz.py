import numpy as np
import argparse
from pathlib import Path

def parse_arguments():
    """Parse command-line arguments for input and output paths."""
    parser = argparse.ArgumentParser(description="Combine two .npz datasets into a single .npz file.")
    parser.add_argument(
        "--input1",
        type=str,
        required=True,
        help="Path to the first .npz dataset (e.g., validation dataset)"
    )
    parser.add_argument(
        "--input2",
        type=str,
        required=True,
        help="Path to the second .npz dataset (e.g., training dataset)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_click.npz",
        help="Path to save the combined .npz dataset (default: combined_click.npz)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output to print dataset details"
    )
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Convert input paths to Path objects for better path handling
    input1_path = Path(args.input1)
    input2_path = Path(args.input2)
    output_path = Path(args.output)

    # Validate input files exist
    if not input1_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input1_path}")
    if not input2_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input2_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the datasets
    if args.verbose:
        print(f"Loading first dataset from: {input1_path}")
    data1 = np.load(input1_path, allow_pickle=True)

    if args.verbose:
        print(f"Loading second dataset from: {input2_path}")
    data2 = np.load(input2_path, allow_pickle=True)

    # Validate that both datasets have the same keys
    if set(data1.files) != set(data2.files):
        raise ValueError("Datasets have different keys. Cannot combine.")

    # Combine the datasets by their keys
    combined_data = {}
    for key in data1.files:
        if args.verbose:
            print(f"Combining data for key: {key}")
        combined_data[key] = np.concatenate((data1[key], data2[key]), axis=0)

    # Save the combined dataset to a new .npz file
    np.savez(output_path, **combined_data)
    print(f"Combined dataset saved at: {output_path}")

    # Verify the saved file
    if args.verbose:
        saved_data = np.load(output_path, allow_pickle=True)
        print(f"Number of keys in the saved file: {len(saved_data.files)}")
        for key in saved_data:
            print(f"Key: {key}, Length of Array: {len(saved_data[key])}")

if __name__ == "__main__":
    main()