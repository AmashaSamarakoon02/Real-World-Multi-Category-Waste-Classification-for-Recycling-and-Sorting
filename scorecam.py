"""
scorecam.py
===========

Score-CAM (Score-Weighted Class Activation Mapping) implementation used for
model explainability (Section 3.1 / Objective 5 of the proposal).

Shared module, following the same pattern as config.py / dataset.py / models.py:
  - scorecam.ipynb imports this module to run the standalone Score-CAM demo.
  - classify_tool.ipynb imports this module for the "Show ScoreCAM" GUI feature.
"""

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import matplotlib.cm as cm


class ScoreCAM:
    """Generates a Score-CAM heatmap for a given model / target layer."""

    def __init__(self, model, target_layer, device):

        self.model = model.eval().to(device)

        self.device = device

        self.activations = None

        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):

        self.activations = out.detach()

    @torch.no_grad()
    def generate(
            self,
            input_tensor,
            class_idx=None,
            batch_size=16):

        input_tensor = input_tensor.to(self.device)

        _, _, H, W = input_tensor.shape

        logits = self.model(input_tensor)

        if class_idx is None:

            class_idx = logits.argmax(dim=1).item()

        activations = self.activations[0]

        num_maps = activations.shape[0]

        upsampled = F.interpolate(
            activations.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        maps_min = upsampled.view(
            num_maps,
            -1
        ).min(dim=1)[0].view(-1, 1, 1)

        maps_max = upsampled.view(
            num_maps,
            -1
        ).max(dim=1)[0].view(-1, 1, 1)

        norm_maps = (
            upsampled - maps_min
        ) / (
            maps_max - maps_min
        ).clamp(min=1e-8)

        weights = torch.zeros(
            num_maps,
            device=self.device
        )

        for start in range(0, num_maps, batch_size):

            end = min(start + batch_size, num_maps)

            masks = norm_maps[start:end].unsqueeze(1)

            outputs = self.model(
                input_tensor * masks
            )

            scores = F.softmax(
                outputs,
                dim=1
            )[:, class_idx]

            weights[start:end] = scores

        cam = (
            weights.view(-1, 1, 1) *
            F.relu(upsampled)
        ).sum(dim=0)

        cam = F.relu(cam)

        cam = cam - cam.min()

        cam = cam / cam.max().clamp(min=1e-8)

        return cam.cpu().numpy(), class_idx


def overlay_heatmap(
        image,
        cam,
        alpha=0.45):
    """Overlays a Score-CAM heatmap (0..1 float array) onto a PIL image."""

    cam = np.array(
        Image.fromarray(
            (cam * 255).astype(np.uint8)
        ).resize(image.size)
    )

    heatmap = cm.jet(
        cam / 255
    )[:, :, :3]

    original = (
        np.array(image)
        .astype(np.float32)
        / 255
    )

    overlay = (
        (1 - alpha) * original +
        alpha * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    return overlay
