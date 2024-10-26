from utils.arg_validator import ArgValidator
from utils.config_validator import ConfigValidator
from utils.dataset_factory import DatasetFactory
from utils.device_utils import get_device


def main():
    # Step 1: Parse and validate the command-line arguments
    config_path = ArgValidator.parse_and_validate_args()

    # Step 2: Load, parse, and validate the configuration file
    config_validator = ConfigValidator(config_path)
    config = config_validator.validate_config()

    print("Configuration file validated successfully: ", config)

    # Step 3: Initialize DatasetFactory with config data
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    data_dir = dataset_config["data_dir"]
    download = dataset_config.get("download", False)

    dataset_factory = DatasetFactory(dataset_name, data_dir, download)

    # Step 4: Load training dataset
    train_dataset = dataset_factory.get_dataset(train=True)
    print("Dataset loaded successfully:", train_dataset)

    # Step 5: Load test dataset
    test_dataset = dataset_factory.get_dataset(train=False)
    print("Dataset loaded successfully:", test_dataset)


if __name__ == "__main__":
    main()
