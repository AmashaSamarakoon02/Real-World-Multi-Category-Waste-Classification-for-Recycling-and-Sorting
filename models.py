"""
models.py
=========

Builds the ImageNet-pretrained CNN backbones used in the project.

Supported models:
    - ResNet50
    - EfficientNet-B0
    - MobileNetV3-Large
"""

import torch.nn as nn
from torchvision import models


def build_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True
):
    """
    Build a CNN model with a new classifier head.

    Returns:
        model
        target_layer (used for Score-CAM)
    """

    backbone_name = backbone_name.lower()

    # -------------------------------------------------
    # ResNet50
    # -------------------------------------------------

    if backbone_name == "resnet50":

        weights = (
            models.ResNet50_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.resnet50(
            weights=weights
        )

        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

        target_layer = model.layer4[-1]

    # -------------------------------------------------
    # EfficientNet-B0
    # -------------------------------------------------

    elif backbone_name == "efficientnet_b0":

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

        target_layer = model.features[-1]

    # -------------------------------------------------
    # MobileNetV3-Large
    # -------------------------------------------------

    elif backbone_name == "mobilenet_v3_large":

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )

        model = models.mobilenet_v3_large(
            weights=weights
        )

        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            num_classes
        )

        target_layer = model.features[-1]

    else:

        raise ValueError(
            f"Unknown backbone '{backbone_name}'.\n"
            "Choose one of:\n"
            "- resnet50\n"
            "- efficientnet_b0\n"
            "- mobilenet_v3_large"
        )

    return model, target_layer


def freeze_backbone(
    model,
    backbone_name: str
):
    """
    Freeze all layers except the classifier head.
    Useful for transfer learning.
    """

    # Freeze everything
    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone_name = backbone_name.lower()

    if backbone_name == "resnet50":

        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    elif backbone_name == "efficientnet_b0":

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    elif backbone_name == "mobilenet_v3_large":

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    else:

        raise ValueError(
            f"Unknown backbone '{backbone_name}'."
        )

    return model