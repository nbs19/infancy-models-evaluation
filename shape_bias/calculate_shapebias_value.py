import os
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
from load_pretrained_models import load_model


def baseline_model(checkpoint_path):
    """Load the baseline model with a similar structure as the current model."""
    # Load your checkpoint for the baseline model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint = checkpoint['state_dict']
    
    # Create the ResNet50 model
    model = models.resnet50(pretrained=False)
    # Adjust the output layer to match the current model
    model.fc = torch.nn.Linear(2048, 1000)  # 1000 classes
    
    # Filter backbone weights (same as current model)
    backbone_state_dict = {
        k.replace('model.backbone.', ''): v 
        for k, v in checkpoint.items() 
        if k.startswith('model.backbone.')
    }
    
    # Load backbone
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier weights (shape should be [1000, 2048])
    if checkpoint['classification_head.weight'].shape == (1000, 2048):
        model.fc.weight.data = checkpoint['classification_head.weight']
        model.fc.bias.data = checkpoint['classification_head.bias']
    
    return model


def extract_texture_class(filename):
    """
    Extracts the texture class from a filename like 'airplane1-bicycle2.png'.
    Returns the texture class name (e.g., 'bicycle').
    """
    parts = filename.split('-')          # Split into ['airplane1', 'bicycle2.png']
    texture_part = parts[1]              # Take 'bicycle2.png'
    texture_part = os.path.splitext(texture_part)[0]  # Remove '.png' -> 'bicycle2'
    for i, char in enumerate(texture_part):
        if char.isdigit():
            return texture_part[:i]      # Return letters before the first digit
    return texture_part  # Fallback if no digit (unlikely)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate shape bias on cue-conflict dataset')
    parser.add_argument('--checkpoint', 
                       type=str, 
                       required=True,
                       help='Path to the model checkpoint file')
    parser.add_argument('--dataset-dir', 
                       type=str, 
                       default="/home/bhargava/shape_bias/model-vs-human/model-vs-human/datasets/cue-conflict/",
                       help='Path to the cue-conflict dataset directory')
    
    args = parser.parse_args()
    
    # Load the model with the specified checkpoint
    model = baseline_model(args.checkpoint)
    model.eval()

    # Define the image transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),                    # Resize to 256x256
        transforms.CenterCrop(224),                # Crop to 224x224 (ResNet50 input size)
        transforms.ToTensor(),                     # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
    ])

    # Load the dataset
    dataset = ImageFolder(root=args.dataset_dir, transform=transform)

    # Get the list of shape class names from subdirectories
    class_names = dataset.classes  # e.g., ['airplane', 'chair', ...]

    # Initialize the mapping from ImageNet probabilities to 16 classes
    mapping = ImageNetProbabilitiesTo16ClassesMapping()

    # Initialize counters for correct decisions
    correct_shape = 0
    correct_texture = 0

    # Process each image in the dataset
    for i, (image, label) in enumerate(dataset):
        # Get the shape class from the subdirectory
        shape_class = class_names[label]  # e.g., 'airplane'

        # Get the filename to extract the texture class
        image_path = dataset.samples[i][0]  # Full path to the image
        filename = os.path.basename(image_path)  # e.g., 'airplane1-bicycle2.png'
        texture_class = extract_texture_class(filename)  # e.g., 'bicycle'

        # Exclude images without cue conflict (shape == texture)
        if shape_class == texture_class:
            continue

        # Classify the image with ResNet50
        with torch.no_grad():
            output = model(image.unsqueeze(0))  # Add batch dimension: [1, C, H, W]
            softmax_output = torch.softmax(output, dim=1).squeeze().cpu().numpy()  # Convert to probabilities

        # Map the softmax output to one of the 16 classes
        decision = mapping.probabilities_to_decision(softmax_output)  # e.g., 'airplane'

        # Check if the prediction matches the shape or texture class
        if decision == shape_class:
            correct_shape += 1
        elif decision == texture_class:
            correct_texture += 1
        # If neither matches, the image is not included in the subset

    # Compute the shape bias
    total_correct = correct_shape + correct_texture
    if total_correct > 0:
        shape_bias = correct_shape / total_correct
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Shape bias: {shape_bias:.4f}")
        print(f"Correct shape decisions: {correct_shape}")
        print(f"Correct texture decisions: {correct_texture}")
        print(f"Total correct decisions: {total_correct}")
    else:
        print("No images were correctly classified as either shape or texture.")


if __name__ == "__main__":
    main()