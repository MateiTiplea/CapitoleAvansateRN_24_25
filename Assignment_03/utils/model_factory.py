# utils/model_factory.py
import importlib.util
import os

from models.lenet import LeNet
from models.mlp import MLP
from models.preact_resnet import PreActResNet18
from models.resnet_timm import ResNet18_CIFAR10


class ModelFactory:
    def __init__(self, config):
        """
        Initializes the ModelFactory with configuration.

        Args:
            config (dict): Configuration dictionary with model specifications.
        """
        self.config = config["training"]
        self.dataset_info = config["dataset"]

    def get_model(self):
        """
        Returns an instance of the specified model.

        Returns:
            nn.Module: The specified model instance.
        """
        model_name = self.config.get("model")
        model_file = self.config.get("model_file")
        model_class = self.config.get("model_class")

        # Load default models based on the model name
        if model_name == "MLP":
            return MLP(num_classes=10)
        elif model_name == "LeNet":
            return LeNet(num_classes=10)
        elif model_name == "resnet18" and self.dataset_info["name"] == "CIFAR10":
            return ResNet18_CIFAR10(num_classes=10)
        elif model_name == "resnet18" and self.dataset_info["name"] == "CIFAR100":
            return ResNet18_CIFAR10(num_classes=100)
        elif model_name == "PreActResNet18" and self.dataset_info["name"] == "CIFAR10":
            return PreActResNet18(num_classes=10)
        elif model_name == "PreActResNet18" and self.dataset_info["name"] == "CIFAR100":
            return PreActResNet18(num_classes=100)

        # Load custom model if specified
        elif model_file and model_class:
            return self._load_custom_model(model_file, model_class)

        else:
            raise ValueError(
                f"Model '{model_name}' is not recognized or properly specified."
            )

    def _load_custom_model(self, model_file, model_class):
        """
        Dynamically loads a model class from a specified Python file.

        Args:
            model_file (str): Path to the Python file containing the model class.
            model_class (str): Name of the model class in the file.

        Returns:
            nn.Module: An instance of the loaded model class.
        """
        if not os.path.isfile(model_file):
            raise ValueError(f"The specified model file '{model_file}' does not exist.")

        # Dynamically load the specified module
        spec = importlib.util.spec_from_file_location("custom_model_module", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Retrieve and instantiate the model class
        if hasattr(module, model_class):
            model_cls = getattr(module, model_class)
            return model_cls()
        else:
            raise ValueError(
                f"The class '{model_class}' was not found in '{model_file}'."
            )
