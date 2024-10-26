# models/mlp.py
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=28 * 28, num_classes=10, hidden_dims=[512, 256, 128]):
        """
        Initializes a simple Multi-Layer Perceptron (MLP) model.

        Args:
            input_dim (int): The input dimension (default 28*28 for MNIST).
            num_classes (int): The number of output classes.
            hidden_dims (list of int): List containing the dimensions of each hidden layer.
        """
        super(MLP, self).__init__()
        self.input_dim = input_dim

        # Define layers
        layers = []
        previous_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout layer for regularization
            previous_dim = dim

        # Output layer
        layers.append(nn.Linear(previous_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the MLP model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        x = x.view(-1, self.input_dim)  # Flatten the input
        return self.model(x)
