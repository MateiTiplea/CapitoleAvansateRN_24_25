import importlib
import os

import yaml


class ConfigValidator:
    REQUIRED_SECTIONS = ["dataset", "training", "output", "dataloader"]
    DEFAULT_DATASETS = {"MNIST", "CIFAR10", "CIFAR100"}
    SUPPORTED_MODELS = {
        "CIFAR10": ["resnet18_cifar10", "PreActResNet18"],
        "CIFAR100": ["resnet18_cifar10", "PreActResNet18"],
        "MNIST": ["MLP", "LeNet"],
    }
    DEFAULT_DATALOADER_PARAMS = {
        "num_workers": 0,
        "batch_size": 64,
        "drop_last": False,
        "persistent_workers": False,
        "shuffle": False,
        "pin_memory": False,
    }
    SUPPORTED_OPTIMIZERS = {"SGD", "Adam", "AdamW", "RMSprop"}
    DEFAULT_OPTIMIZER_PARAMS = {
        "SGD": {"lr": 0.01, "momentum": 0.0, "weight_decay": 0.0, "nesterov": False},
        "Adam": {"lr": 0.001, "weight_decay": 0.0},
        "AdamW": {"lr": 0.001, "weight_decay": 0.0},
        "RMSprop": {"lr": 0.01, "momentum": 0.0, "weight_decay": 0.0},
    }

    def __init__(self, config_path):
        self.config_path = config_path
        self.config_data = self._load_config()

    def _load_config(self):
        """Loads the YAML configuration file."""
        try:
            with open(self.config_path, "r") as file:
                config_data = yaml.safe_load(file)
            if config_data is None:
                raise ValueError("The configuration file is empty.")
            return config_data
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")

    def validate_config(self):
        """Validates the presence of required sections and dataset attributes."""
        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in self.config_data:
                raise ValueError(
                    f"Missing required section: '{section}' in the config file."
                )

        # Validate dataset section
        dataset_section = self.config_data.get("dataset")
        if "name" not in dataset_section:
            raise ValueError("The 'dataset' section must contain a 'name' attribute.")

        dataset_name = dataset_section["name"]
        self._validate_data_dir(dataset_section)
        self._set_or_validate_download_flag(dataset_section, dataset_name)

        if dataset_name not in self.DEFAULT_DATASETS:
            self._validate_custom_dataset(dataset_section)

        # Validate transform scripts
        self._validate_transform_scripts(dataset_section)

        # Validate dataloader configuration
        self._validate_dataloader()

        # Validate training configuration including model selection and optimizer
        self._validate_training(dataset_name)

        return self.config_data

    def _validate_data_dir(self, dataset_section):
        """Validates the data_dir attribute, ensuring it's a valid directory or creatable."""
        if "data_dir" not in dataset_section:
            raise ValueError(
                "The 'dataset' section must contain a 'data_dir' attribute."
            )

        data_dir = dataset_section["data_dir"]
        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                print(f"Created data directory at '{data_dir}' as it did not exist.")
            except Exception as e:
                raise ValueError(
                    f"Could not create the specified data directory '{data_dir}': {e}"
                )
        elif not os.path.isdir(data_dir):
            raise ValueError(
                f"The specified data directory '{data_dir}' exists but is not a directory."
            )

    def _set_or_validate_download_flag(self, dataset_section, dataset_name):
        """Sets or validates the download flag for default datasets based on the data_dir content."""
        download = dataset_section.get("download", False)

        if dataset_name in self.DEFAULT_DATASETS:
            data_dir = dataset_section["data_dir"]
            if os.path.isdir(data_dir) and not os.listdir(data_dir):
                download = True
                print(
                    f"The data directory '{data_dir}' is empty. Setting 'download' to True."
                )

        if not isinstance(download, bool):
            raise ValueError(
                "The 'download' attribute in the 'dataset' section must be a boolean (True or False)."
            )

        dataset_section["download"] = (
            download  # Update the config data with the final download value
        )

    def _validate_custom_dataset(self, dataset_section):
        """Validates custom dataset attributes such as module path, class, and data_dir."""
        dataset_name = dataset_section["name"]

        if "." not in dataset_name:
            raise ValueError(
                "Custom dataset name must be in '<module_name>.<dataset_class>' format."
            )

        module_name, class_name = dataset_name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, class_name):
                raise ImportError(
                    f"The specified class '{class_name}' was not found in module '{module_name}'."
                )
        except ImportError as e:
            raise ImportError(f"Could not import custom dataset '{dataset_name}': {e}")

    def _validate_transform_scripts(self, dataset_section):
        """Validates the presence and readability of transform script paths if specified."""
        for script_key in ["initial_transform_script", "runtime_transform_script"]:
            script_path = dataset_section.get(script_key)
            if script_path:
                if not os.path.isfile(script_path):
                    raise ValueError(
                        f"The specified transform script '{script_path}' does not exist."
                    )
                if not script_path.endswith(".py"):
                    raise ValueError(
                        f"The transform script '{script_path}' must be a Python (.py) file."
                    )
                print(f"Transform script '{script_path}' validated successfully.")

    def _validate_dataloader(self):
        """Validates the dataloader section with train and test configurations."""
        dataloader_section = self.config_data.get("dataloader")
        for mode in ["train", "test"]:
            if mode not in dataloader_section:
                raise ValueError(
                    f"Missing '{mode}' configuration in the 'dataloader' section."
                )

            # Validate each dataloader parameter and set defaults where necessary
            dataloader_config = dataloader_section[mode]
            for param, default_value in self.DEFAULT_DATALOADER_PARAMS.items():
                if param in dataloader_config:
                    # Ensure correct type for each parameter
                    if param in ["num_workers", "batch_size"] and not isinstance(
                        dataloader_config[param], int
                    ):
                        raise ValueError(
                            f"'{param}' in dataloader '{mode}' must be an integer."
                        )
                    if param in [
                        "drop_last",
                        "persistent_workers",
                        "shuffle",
                        "pin_memory",
                    ] and not isinstance(dataloader_config[param], bool):
                        raise ValueError(
                            f"'{param}' in dataloader '{mode}' must be a boolean."
                        )
                else:
                    # Set default if parameter is missing
                    dataloader_config[param] = default_value

    def _validate_training(self, dataset_name):
        """Validates the training section, focusing on model specification."""
        training_section = self.config_data.get("training")

        # Validate model selection
        self._validate_model(dataset_name, training_section)

        # Validate optimizer selection
        self._validate_optimizer(training_section)

    def _validate_model(self, dataset_name, training_section):
        model_name = training_section.get("model")
        model_file = training_section.get("model_file")
        model_class = training_section.get("model_class")

        if model_name:
            if model_name not in self.SUPPORTED_MODELS.get(dataset_name, []):
                raise ValueError(
                    f"Model '{model_name}' is not supported for dataset '{dataset_name}'."
                )

        elif model_file and model_class:
            if not os.path.isfile(model_file):
                raise ValueError(
                    f"The specified model file '{model_file}' does not exist."
                )
            if not model_file.endswith(".py"):
                raise ValueError(
                    f"The model file '{model_file}' must be a Python (.py) file."
                )
            if not model_class:
                raise ValueError(
                    "The 'model_class' must be specified when using a custom model file."
                )

        else:
            raise ValueError(
                "You must specify either 'model' or both 'model_file' and 'model_class' in the training section."
            )

    def _validate_optimizer(self, training_section):
        """Validates the optimizer section within training."""
        optimizer_section = training_section.get("optimizer")
        if not optimizer_section:
            raise ValueError(
                "The 'training' section must contain an 'optimizer' subsection."
            )

        optimizer_name = optimizer_section.get("name")
        if optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: '{optimizer_name}'.")

        # Validate optimizer parameters, setting defaults where necessary
        params = optimizer_section.get("params", {})
        default_params = self.DEFAULT_OPTIMIZER_PARAMS[optimizer_name]

        for param, default_value in default_params.items():
            if param in params:
                if param in ["lr", "weight_decay", "momentum"] and not isinstance(
                    params[param], (int, float)
                ):
                    raise ValueError(
                        f"'{param}' for optimizer '{optimizer_name}' must be a number."
                    )
                if param == "nesterov" and not isinstance(params[param], bool):
                    raise ValueError(
                        f"'nesterov' for optimizer '{optimizer_name}' must be a boolean."
                    )
            else:
                # Set default if parameter is missing
                params[param] = default_value

        optimizer_section["params"] = params  # Update config with validated params
        print(f"Optimizer '{optimizer_name}' validated successfully.")
