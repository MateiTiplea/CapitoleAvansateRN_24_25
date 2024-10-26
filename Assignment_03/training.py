import torch
import torch.nn as nn
import torch.optim as optim

from utils.arg_validator import ArgValidator
from utils.config_validator import ConfigValidator
from utils.device_utils import get_device


# Dummy model for illustration (Replace with a real model later)
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(784, 10)  # Example linear layer (for MNIST)

    def forward(self, x):
        return self.fc(x)


def train():
    # Step 1: Get the device (CPU or GPU)
    device = get_device()
    print(f"Using device: {device}")

    # Step 2: Initialize model and move it to the device
    model = DummyModel().to(device)

    # Step 3: Define a dummy dataset and DataLoader (Replace with actual dataset)
    x = torch.randn(64, 784)  # Dummy data
    y = torch.randint(0, 10, (64,))  # Dummy labels

    # Step 4: Move data to the same device as the model
    x, y = x.to(device), y.to(device)

    # Define loss function and optimizer (for illustration)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Dummy training loop
    model.train()
    for epoch in range(2):  # Example of 2 epochs
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/2], Loss: {loss.item()}")


def main():
    # Step 1: Parse and validate the command-line arguments
    config_path = ArgValidator.parse_and_validate_args()

    # Step 2: Load, parse, and validate the configuration file
    config_validator = ConfigValidator(config_path)
    config = config_validator.validate_config()

    print("Configuration file validated successfully: ", config)


if __name__ == "__main__":
    main()
