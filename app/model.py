from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class HybridClassifier(nn.Module):
    """EfficientNet image embedding fused with projected handcrafted descriptors."""

    def __init__(self, num_classes: int = 5, handcrafted_dim: int = 1821, pretrained: bool = False):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_projection = nn.Sequential(nn.Flatten(), nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.3))
        self.handcrafted_projection = nn.Sequential(nn.Linear(handcrafted_dim, 128), nn.BatchNorm1d(128),
                                                     nn.ReLU(), nn.Dropout(0.25))
        self.gate = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.Linear(384, 192), nn.ReLU(), nn.Dropout(0.4),
                                        nn.Linear(192, num_classes))
        # Freeze early feature stages while allowing later convolutional stages to fine-tune.
        for layer in list(self.features.children())[:-3]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor) -> torch.Tensor:
        cnn = self.cnn_projection(self.pool(self.features(image)))
        hand = self.handcrafted_projection(handcrafted)
        joined = torch.cat((cnn, hand), dim=1)
        weights = self.gate(joined)
        return self.classifier(torch.cat((cnn * weights[:, :1], hand * weights[:, 1:]), dim=1))
