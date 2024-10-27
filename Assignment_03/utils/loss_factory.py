# utils/loss_factory.py
import torch.nn as nn


class LossFactory:
    def __init__(self, config):
        """
        Initializes the LossFactory with configuration.

        Args:
            config (dict): Configuration dictionary containing the loss specification.
        """
        self.loss_name = config["training"].get("loss")

    def get_loss_function(self):
        """
        Returns an instance of the specified loss function.

        Returns:
            nn.Module: The specified loss function instance.
        """
        if self.loss_name == "CrossEntropyLoss":
            return nn.CrossEntropyLoss()
        elif self.loss_name == "MSELoss":
            return nn.MSELoss()
        else:
            raise ValueError(f"Loss function '{self.loss_name}' is not supported.")
