import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from grad_cam import GradCAM
from UnetNested.Nested_Unet import NestedUNet


# Function to visualize the original .npy file (CT slice)
def visualize_npy(file_path):
    # Load the .npy file
    data = np.load(file_path)

    # Normalize the image for better visualization
    data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize to [0, 1]
    plt.imshow(data, cmap="gray")
    plt.title("Original CT Slice")
    plt.colorbar()
    plt.show()


def apply_grad_cam(file_path, model_path, save_path):
    # Load your trained model
    model = NestedUNet(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'), weights_only=True))
    model.eval().cuda()

    # Use the deepest encoder layer `conv4_0` for Grad-CAM
    target_layer = model.conv3_0

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Load the input CT slice
    sample_input = np.load(file_path)

    # Ensure the input has the correct shape for your model
    if sample_input.ndim == 2:  # Add a channel dimension if missing
        sample_input = sample_input[np.newaxis, ...]
    sample_input = torch.tensor(sample_input).unsqueeze(0).cuda()  # Add batch dimension
    sample_input = sample_input.float()  # Convert to torch.float32

    # Debugging input shape
    print("Input shape:", sample_input.shape)

    # Generate the model's prediction (soft mask)
    with torch.no_grad():
        prediction = model(sample_input)
    print("Prediction shape:", prediction.shape)

    # Apply Grad-CAM to the target layer
    heatmap = grad_cam.generate(sample_input, class_idx=0)  # Set class_idx to 0 for binary segmentation

    # Debugging heatmap
    print("Heatmap shape:", heatmap.shape)
    print("Heatmap range:", heatmap.min(), heatmap.max())

    # Save the Grad-CAM heatmap overlaid on the original CT slice
    original_image = sample_input.squeeze().cpu().numpy()

    # Plot and save the heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap with transparency
    plt.colorbar()
    plt.title("Grad-CAM Heatmap")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the heatmap to the specified path
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.close()  # Close the plot to free memory

    # Debugging: Visualize the input tensor
    print("Sample input shape:", sample_input.shape)
    print("Sample input range:", sample_input.min().item(), sample_input.max().item())

    # Debugging: Check model predictions
    print("Prediction shape:", prediction.shape)
    print("Prediction range:", prediction.min().item(), prediction.max().item())



if __name__ == "__main__":
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Define the paths
    model_name = "NestedUNET_with_augmentation"
    file_name = "0023_NI000_slice002.npy"  # Original CT slice
    model_file_name = "model.pth"  # Trained model weights
    output_file_name = "grad_cam_heatmap.png"  # Heatmap output file name

    # Construct full paths
    file_path = os.path.join("../Preprocessing/data/Image/LIDC-IDRI-0023", file_name)  # Adjusted for sibling folder
    model_path = os.path.join("model_outputs", model_name, model_file_name)
    save_path = os.path.join("grad_cam_outputs", output_file_name)  # Save directory for heatmaps

    # Debugging: Print paths for validation
    print(f"File path for input: {file_path}")
    print(f"File path for model: {model_path}")
    print(f"Heatmap save path: {save_path}")

    # Ensure paths are correct
    if not os.path.exists(file_path):
        print(f"Error: Input file {file_path} not found.")
        exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        exit(1)

    # Apply Grad-CAM on the model's prediction and save the heatmap
    print(f"Applying Grad-CAM on {file_path}")
    apply_grad_cam(file_path, model_path, save_path)
