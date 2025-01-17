import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from UnetNested.Nested_Unet import NestedUNet

import cv2

def apply_grad_cam_with_library(file_path, model_path, save_path):
    # Load the trained model
    model = NestedUNet(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'), weights_only=True))
    model.eval().cuda()

    # Specify the target layer
    target_layer = model.conv2_0  # Replace with the desired layer

    # Load the input CT slice
    sample_input = np.load(file_path)

    # Normalize input and add batch dimension
    sample_input = (sample_input - np.min(sample_input)) / (np.max(sample_input) - np.min(sample_input))  # Normalize to [0, 1]
    sample_input = torch.tensor(sample_input[np.newaxis, np.newaxis, ...], dtype=torch.float32).cuda()  # Add batch dimension

    # Create a dummy mask with the same spatial dimensions as the input
    mask = np.ones(sample_input.shape[2:], dtype=np.float32)  # A full mask (all ones)

    # Define the Grad-CAM target
    targets = [SemanticSegmentationTarget(0, mask)]  # Target the segmentation class index 0

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate the Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=sample_input, targets=targets)[0]  # For batch size of 1

    # Prepare the original image for visualization
    sample_input_np = sample_input.squeeze().cpu().numpy()
    sample_input_np_rgb = cv2.cvtColor(sample_input_np, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB

    # Overlay the heatmap on the original image
    visualization = show_cam_on_image(sample_input_np_rgb, grayscale_cam, use_rgb=True)

    # Save the heatmap
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, visualization)
    print(f"Heatmap saved to {save_path}")

# Main function
if __name__ == "__main__":
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Define the paths
    model_name = "NestedUNET_with_augmentation"
    file_name = "0021_NI002_slice003.npy"  # Original CT slice
    model_file_name = "model.pth"  # Trained model weights
    output_file_name = "grad_cam_heatmap_library.png"  # Heatmap output file name
    

    # Construct full paths
    file_path = os.path.join("../Preprocessing/data/Image/LIDC-IDRI-0021", file_name)  # Adjusted for sibling folder
    model_path = os.path.join("model_outputs", model_name, model_file_name)
    save_path = os.path.join("grad_cam_outputs", output_file_name)  # Save directory for heatmaps

    # Ensure paths are correct
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} not found.")
        exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        exit(1)

    # Apply Grad-CAM on the model's prediction and save the heatmap
    print(f"Applying Grad-CAM on {file_path}")
    apply_grad_cam_with_library(file_path, model_path, save_path)
