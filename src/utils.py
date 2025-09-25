from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def tensor_to_pil(t: torch.Tensor, denorm: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    if denorm is not None:
        mean, std = denorm
        for c in range(3):
            t[c] = t[c] * std[c] + mean[c]
    t = t.clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(t.cpu())

def plot_images(images: torch.Tensor, titles: Optional[list] = None, cols: int = 4, denorm: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None) -> None:
    n = images.size(0)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = tensor_to_pil(images[i], denorm=denorm)
        plt.imshow(img)
        if titles:
            plt.title(str(titles[i]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def grad_cam(model: torch.nn.Module, img_tensor: torch.Tensor, target_layer: str, class_idx: Optional[int] = None) -> np.ndarray:
    model.eval()
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    layer = dict(model.named_modules())[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_backward_hook(bwd_hook)

    try:
        img = img_tensor if img_tensor.dim() == 4 else img_tensor.unsqueeze(0)
        img = img.requires_grad_(True)
        logits = model(img)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx].sum()
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        act = activations["value"]        # [B, C, H, W]
        grad = gradients["value"]         # [B, C, H, W]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * act).sum(dim=1)  # [B, H, W]
        cam = torch.relu(cam)
        cam = cam / (cam.max(dim=(1, 2), keepdim=True).values + 1e-6)
        cam_np = cam[0].cpu().numpy()
        return cam_np
    finally:
        h1.remove()
        h2.remove()

def compute_classification_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        pred_labels = preds.argmax(dim=1)
        correct = (pred_labels == targets).sum().item()
        total = targets.size(0)
        acc = correct / total if total > 0 else 0.0
        return {"accuracy": acc}
