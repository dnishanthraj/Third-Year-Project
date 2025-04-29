# ---------------------------------------------------------
# Grad-CAM (Gradient-weighted Class Activation Mapping)
# ---------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

# ---------------------------------------------------------
# GradCAM Class Definition
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM object.

        Args:
            model (torch.nn.Module): The trained model for which Grad-CAM is to be generated.
            target_layer (torch.nn.Module): The specific convolutional layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Register hooks to capture forward activations and backward gradients
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """
        Saves the forward activation output during the forward pass.
        """
        self.activation = output

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Saves the gradients during the backward pass.
        """
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        """
        Generates the Grad-CAM heatmap for a given input.

        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W).
            class_idx (int, optional): Class index to generate Grad-CAM for. If None, uses the argmax output.

        Returns:
            np.ndarray: Normalized Grad-CAM heatmap.
        """
        self.model.eval()  # Set model to evaluation mode
        output = self.model(input_tensor)

        # Choose class index: default to predicted class if not specified
        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backpropagate gradients for the selected class
        self.model.zero_grad()
        scalar_output = output[:, class_idx].mean()  # Reduce to scalar value
        scalar_output.backward()

        # Compute Grad-CAM: average gradients spatially to obtain weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        grad_cam = torch.sum(weights * self.activation, dim=1).squeeze(0)  # Weighted sum across channels

        # Apply ReLU to retain only positive influences
        grad_cam = F.relu(grad_cam)

        # Convert Grad-CAM to NumPy array
        grad_cam = grad_cam.cpu().detach().numpy()

        # Resize heatmap to match input size (H x W)
        grad_cam = cv2.resize(grad_cam, (input_tensor.shape[2], input_tensor.shape[3]))

        # Normalize heatmap to [0, 1]
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

        # -------------------------------
        # Debugging (Optional Prints)
        # -------------------------------
        # print("Activation shape:", self.activation.shape)
        # print("Gradient shape:", self.gradients.shape)
        # print("Weights shape:", weights.shape)
        # print("Grad-CAM shape:", grad_cam.shape)

        return grad_cam
