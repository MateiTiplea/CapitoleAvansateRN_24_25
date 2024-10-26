import importlib
import os

import yaml


class ConfigValidator:
    REQUIRED_SECTIONS = ["dataset", "training", "output"]
    DEFAULT_DATASETS = {"MNIST", "CIFAR10", "CIFAR100"}

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

        # Validate the dataset section
        dataset_section = self.config_data.get("dataset")
        if "name" not in dataset_section:
            raise ValueError("The 'dataset' section must contain a 'name' attribute.")

        dataset_name = dataset_section["name"]

        # Ensure data_dir is present and valid
        self._validate_data_dir(dataset_section)

        # Set or validate the download flag based on data_dir content
        self._set_or_validate_download_flag(dataset_section, dataset_name)

        # Check if the dataset name is one of the defaults or custom
        if dataset_name not in self.DEFAULT_DATASETS:
            self._validate_custom_dataset(dataset_section)

        # Validate transform scripts
        self._validate_transform_scripts(dataset_section)

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

        # If the dataset is default and data_dir is empty, set download to True
        if dataset_name in self.DEFAULT_DATASETS:
            data_dir = dataset_section["data_dir"]
            if os.path.isdir(data_dir) and not os.listdir(data_dir):
                # Directory is empty, automatically set download to True
                download = True
                print(
                    f"The data directory '{data_dir}' is empty. Setting 'download' to True."
                )

        # Ensure download is a boolean
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

        # Validate custom dataset format
        if "." not in dataset_name:
            raise ValueError(
                "Custom dataset name must be in '<module_name>.<dataset_class>' format."
            )

        # Split the module and class name
        module_name, class_name = dataset_name.rsplit(".", 1)

        # Attempt to import the module and class
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
