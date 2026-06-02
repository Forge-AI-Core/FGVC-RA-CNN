import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Union


class GradCAMHook:
    """PyTorch forward and backward activation/gradient registration hook.

    Attributes:
        target_layer (nn.Module): Target convolutional layer to inspect.
        activations (torch.Tensor): Captured activations from forward pass.
        gradients (torch.Tensor): Captured gradients from backward pass.
    """

    def __init__(self, target_layer: nn.Module) -> None:
        """Initializes the hook and registers it on target_layer."""
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self.forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        """Forward pass hook function."""
        self.activations = output

    def backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        """Backward pass hook function."""
        self.gradients = grad_output[0]

    def remove(self) -> None:
        """Removes the hook handles from target_layer."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """Finds the last Conv2d layer in the model or model features block.

    Args:
        model (nn.Module): The classification network.

    Returns:
        nn.Module: The last Conv2d layer found.

    Raises:
        ValueError: If no Conv2d layer is found in the model.
    """
    last_conv = None
    features_block = getattr(model, "features", model)

    for sub_module in features_block.modules():
        if isinstance(sub_module, nn.Conv2d):
            last_conv = sub_module

    if last_conv is None:
        raise ValueError("Could not find any nn.Conv2d layer in the model features.")

    return last_conv


def compute_gradcam_plusplus_heatmap(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    device: str,
) -> np.ndarray:
    """Computes the Grad-CAM++ normalized heatmap for a single image and target class.

    Args:
        model (nn.Module): Classifier model.
        target_layer (nn.Module): Convolutional layer to capture feature maps.
        input_tensor (torch.Tensor): Image input tensor of shape [1, 3, H, W].
        class_idx (int): Target class index.
        device (str): Device to run computation on (e.g. 'cuda', 'cpu').

    Returns:
        np.ndarray: Normalized 2D Grad-CAM++ heatmap array of shape [H, W].
    """
    hook = GradCAMHook(target_layer=target_layer)
    
    # Ensure gradient calculations are tracked for input
    x = input_tensor.clone().detach().to(device)
    x.requires_grad = True

    # Forward pass
    logits = model(x)
    model.zero_grad()

    # Backward pass for the score of the specific target class
    score = logits[0, class_idx]
    score.backward()

    # Retrieve activations and gradients
    activations = hook.activations.detach().float()  # Shape: [1, C, H, W]
    gradients = hook.gradients.detach().float()      # Shape: [1, C, H, W]

    # Grad-CAM++ Weight Formulation:
    # w_k = sum_i_j( alpha_i_j * relu(grad_i_j) )
    # alpha_i_j = grad^2 / (2 * grad^2 + sum(act) * grad^3)
    pos_gradients = F.relu(gradients)
    grads_power_2 = gradients ** 2
    grads_power_3 = gradients ** 3

    sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
    
    # Compute the denominator and safeguard against division by zero
    denominator = 2.0 * grads_power_2 + sum_activations * grads_power_3
    denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
    
    alpha = grads_power_2 / denominator
    
    # Compute the weighted gradients
    weights = torch.sum(alpha * pos_gradients, dim=(2, 3), keepdim=True)
    
    # Linear combination of activation maps followed by ReLU
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    cam = F.relu(cam)

    # Upsample to match original resolution
    cam = F.interpolate(
        cam,
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode="bilinear",
        align_corners=False,
    )
    cam = cam.squeeze().cpu().numpy()

    # Normalize to [0.0, 1.0] range safely
    cam_min, cam_max = cam.min(), cam.max()
    epsilon = 1e-8
    if (cam_max - cam_min) > epsilon:
        normalized_cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        normalized_cam = np.zeros_like(cam)

    hook.remove()
    return normalized_cam


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalizes an ImageNet-normalized image tensor back to a [0, 1] RGB numpy array.

    Args:
        tensor (torch.Tensor): Transformed input tensor of shape [3, H, W].

    Returns:
        np.ndarray: Denormalized image array of shape [H, W, 3] in float32.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)

    denorm = tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0)
    
    return denorm.permute(1, 2, 0).cpu().numpy()


def overlay_heatmap_on_image(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Superimposes a colormapped Grad-CAM++ heatmap onto a denormalized image.

    Args:
        image (np.ndarray): HWC RGB float32 image array in range [0, 1].
        heatmap (np.ndarray): HW float32 normalized activation map in range [0, 1].
        alpha (float): Transparency mixing factor between heatmap and image.

    Returns:
        np.ndarray: Overlay composite image of shape [H, W, 3].
    """
    # Create colormap RGB array using matplotlib's jet colormap
    color_heatmap = plt.cm.jet(heatmap)[:, :, :3]
    
    # Blend color heatmap with original image
    blend = alpha * color_heatmap + (1.0 - alpha) * image
    return np.clip(blend, 0.0, 1.0)


def visualize_and_save_gradcam_comparison(
    dataset_name: str,
    images: torch.Tensor,
    crop1: torch.Tensor,
    crop2: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    model: nn.Module,
    device: str,
) -> None:
    """Generates and saves a visual dashboard showing multi-scale crops and Grad-CAM++ overlays.

    Args:
        dataset_name (str): Name of the dataset (for folders).
        images (torch.Tensor): Original images tensor of shape [B, 3, H, W].
        crop1 (torch.Tensor): Stage 1 APN crop images tensor of shape [B, 3, H, W].
        crop2 (torch.Tensor): Stage 2 APN crop images tensor of shape [B, 3, H, W].
        predictions (torch.Tensor): Model ensemble class predictions of shape [B].
        labels (torch.Tensor): Ground truth labels of shape [B].
        class_names (List[str]): List of human-readable class names.
        model (nn.Module): The final RACNN ensemble network.
        device (str): Device to perform backpropagation on.
    """
    num_samples = len(images)
    rows_data = []

    # Target layers to extract feature maps from (last Conv2d of the classifiers)
    target_layer1 = find_last_conv_layer(model.racnn.classifier1)
    target_layer2 = find_last_conv_layer(model.racnn.classifier2)
    target_layer3 = find_last_conv_layer(model.racnn.classifier3)

    # Compute heatmaps sequentially under explicit grad enabling
    with torch.enable_grad():
        for i in range(num_samples):
            pred_class = predictions[i].item()

            # Grad-CAM++ Heatmaps for the 3 scales
            h1 = compute_gradcam_plusplus_heatmap(
                model.racnn.classifier1, target_layer1, images[i].unsqueeze(0), pred_class, device
            )
            h2 = compute_gradcam_plusplus_heatmap(
                model.racnn.classifier2, target_layer2, crop1[i].unsqueeze(0), pred_class, device
            )
            h3 = compute_gradcam_plusplus_heatmap(
                model.racnn.classifier3, target_layer3, crop2[i].unsqueeze(0), pred_class, device
            )

            # Denormalize
            img1_val = denormalize_image(images[i])
            img2_val = denormalize_image(crop1[i])
            img3_val = denormalize_image(crop2[i])

            # Overlay composites
            o1 = overlay_heatmap_on_image(img1_val, h1)
            o2 = overlay_heatmap_on_image(img2_val, h2)
            o3 = overlay_heatmap_on_image(img3_val, h3)

            rows_data.append((img1_val, o1, img2_val, o2, img3_val, o3))

    # Grid plot generation
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    plt.suptitle(f"RA-CNN Grad-CAM++ Multi-Scale Visualizations ({dataset_name})", fontsize=18, y=0.99)

    headers = [
        "Scale 1: Input", "Scale 1: Grad-CAM++",
        "Scale 2: Crop 1", "Scale 2: Grad-CAM++",
        "Scale 3: Crop 2", "Scale 3: Grad-CAM++"
    ]

    for row_idx in range(num_samples):
        data_cols = rows_data[row_idx]
        gt = labels[row_idx].item()
        pred = predictions[row_idx].item()

        gt_name = class_names[gt] if gt < len(class_names) else f"Class {gt}"
        pred_name = class_names[pred] if pred < len(class_names) else f"Class {pred}"
        is_correct = "O" if gt == pred else "X"

        for col_idx in range(6):
            ax = axes[col_idx] if num_samples == 1 else axes[row_idx, col_idx]
            ax.imshow(data_cols[col_idx])
            ax.axis("off")

            # Label on the left side of the row
            if col_idx == 0:
                ax.text(
                    -0.15,
                    0.5,
                    f"No.{row_idx+1}\nGT: {gt_name}\nPred: {pred_name} ({is_correct})",
                    transform=ax.transAxes,
                    fontsize=10,
                    ha="right",
                    va="center",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            # Titles on top row
            if row_idx == 0:
                ax.set_title(headers[col_idx], fontsize=13, pad=10)

    plt.tight_layout()
    
    # Save the output image
    results_dir = Path(f"results/{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "gradcam_comparison.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"📊 Grad-CAM++ multi-scale visualization saved: {save_path}\n")
