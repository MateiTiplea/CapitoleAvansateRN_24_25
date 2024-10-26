# utils/optimizer_factory.py
import torch.optim as optim


class OptimizerFactory:
    def __init__(self, config):
        """
        Initializes the OptimizerFactory with configuration.

        Args:
            config (dict): Configuration dictionary containing optimizer specifications.
        """
        self.config = config["training"]["optimizer"]

    def get_optimizer(self, model_parameters):
        """
        Returns an instance of the specified optimizer.

        Args:
            model_parameters (iterable): Model parameters to be optimized.

        Returns:
            torch.optim.Optimizer: The specified optimizer instance.
        """
        optimizer_name = self.config.get("name")
        params = self.config.get("params", {})

        if optimizer_name == "SGD":
            return optim.SGD(model_parameters, **params)
        elif optimizer_name == "Adam":
            return optim.Adam(model_parameters, **params)
        elif optimizer_name == "AdamW":
            return optim.AdamW(model_parameters, **params)
        elif optimizer_name == "RMSprop":
            return optim.RMSprop(model_parameters, **params)
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")
