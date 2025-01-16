import torch
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient) 

    def _save_activation(self, module, input, output):
        self.activation = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        # Choose the class index for which Grad-CAM is applied
        if class_idx is None:
            class_idx = torch.argmax(output)

        # Calculate gradients for the target class
        self.model.zero_grad()
        scalar_output = output[:, class_idx].mean()  # Reduce to scalar
        scalar_output.backward()

        # Compute weights from gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activation, dim=1).squeeze(0)

        # Apply ReLU to retain positive influences
        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam.cpu().detach().numpy()

        # Normalize the heatmap for visualization
        grad_cam = cv2.resize(grad_cam, (input_tensor.shape[2], input_tensor.shape[3]))
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

        # Debugging: Print activation and gradient shapes
        print("Activation shape:", self.activation.shape)
        print("Gradient shape:", self.gradients.shape)

        # Debugging: Check weights and Grad-CAM tensor
        print("Weights shape:", weights.shape)
        print("Grad-CAM shape:", grad_cam.shape)

        return grad_cam

