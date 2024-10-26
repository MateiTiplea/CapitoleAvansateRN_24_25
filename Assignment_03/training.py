from utils.arg_validator import ArgValidator
from utils.config_validator import ConfigValidator
from utils.dataloader_factory import DataLoaderFactory
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

    # Optional transform scripts
    initial_transform_script = dataset_config.get("initial_transform_script")
    runtime_transform_script = dataset_config.get("runtime_transform_script")

    dataset_factory = DatasetFactory(
        dataset_name,
        data_dir,
        download=download,
        initial_transform_script=initial_transform_script,
        runtime_transform_script=runtime_transform_script,
    )

    # Step 4: Load training dataset
    train_dataset = dataset_factory.get_dataset(train=True)
    print("Dataset loaded successfully:", train_dataset)

    # Step 5: Load test dataset
    test_dataset = dataset_factory.get_dataset(train=False)
    print("Dataset loaded successfully:", test_dataset)

    # Step 6: Initialize DataLoaderFactory with config data
    dataloader_factory = DataLoaderFactory(config["dataloader"])

    # Step 7: Create DataLoaders for training and test datasets
    train_dataloader = dataloader_factory.create_dataloader(train_dataset, mode="train")
    test_dataloader = dataloader_factory.create_dataloader(test_dataset, mode="test")

    print("DataLoaders created successfully.")
    print("Training DataLoader:", train_dataloader)
    print("Test DataLoader:", test_dataloader)


if __name__ == "__main__":
    main()
