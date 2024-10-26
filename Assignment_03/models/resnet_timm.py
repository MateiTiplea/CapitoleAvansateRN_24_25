# models/resnet_timm.py
import timm
import torch.nn as nn


class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        """
        Initializes a ResNet-18 model for CIFAR-10 classification using timm.

        Args:
            num_classes (int): Number of output classes. Default is 10 for CIFAR-10.
            pretrained (bool): Whether to use pretrained weights from timm.
        """
        super(ResNet18_CIFAR10, self).__init__()

        # Load ResNet-18 model from timm
        self.model = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass for ResNet-18.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.model(x)
