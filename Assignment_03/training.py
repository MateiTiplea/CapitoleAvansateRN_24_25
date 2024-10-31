import os

import torch

from utils.arg_validator import ArgValidator
from utils.config_validator import ConfigValidator
from utils.dataloader_factory import DataLoaderFactory
from utils.dataset_factory import DatasetFactory
from utils.device_utils import get_device
from utils.loss_factory import LossFactory
from utils.model_factory import ModelFactory
from utils.optimizer_factory import OptimizerFactory
from utils.scheduler_factory import SchedulerFactory
from utils.trainer import Trainer


def main():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, 0)

    # Step 1: Parse and validate the command-line arguments
    config_path = ArgValidator.parse_and_validate_args()

    # Step 2: Load, parse, and validate the configuration file
    config_validator = ConfigValidator(config_path)
    config = config_validator.validate_config()

    # print("Configuration file validated successfully: ", config)
    print("Configuration file validated successfully.")

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
    # print("Dataset loaded successfully:", train_dataset)
    print("Dataset loaded successfully.")

    # Step 5: Load test dataset
    test_dataset = dataset_factory.get_dataset(train=False)
    # print("Dataset loaded successfully:", test_dataset)
    print("Dataset loaded successfully.")

    # Step 6: Initialize DataLoaderFactory with config data
    dataloader_factory = DataLoaderFactory(config["dataloader"])

    # Step 7: Create DataLoaders for training and test datasets
    train_dataloader = dataloader_factory.create_dataloader(train_dataset, mode="train")
    test_dataloader = dataloader_factory.create_dataloader(test_dataset, mode="test")

    print("DataLoaders created successfully.")
    # print("Training DataLoader:", train_dataloader)
    # print("Test DataLoader:", test_dataloader)

    # Step 8: Initialize ModelFactory with config data
    model_factory = ModelFactory(config)
    model = model_factory.get_model()
    # print("Model loaded successfully:", model)
    print("Model loaded successfully.")

    # Step 9: Initialize OptimizerFactory with config data
    optimizer_factory = OptimizerFactory(config)
    optimizer = optimizer_factory.get_optimizer(model.parameters())
    # print("Optimizer loaded successfully:", optimizer)
    print("Optimizer loaded successfully.")

    # Step 10: Initialize SchedulerFactory and get scheduler (if specified)
    scheduler_factory = SchedulerFactory(config)
    scheduler = scheduler_factory.get_scheduler(optimizer)
    if scheduler:
        # print("Scheduler initialized successfully:", scheduler)
        print("Scheduler initialized successfully.")
    else:
        print("No scheduler is being used for this training.")

    # Step 11: Initialize LossFactory and get loss function
    loss_factory = LossFactory(config)
    criterion = loss_factory.get_loss_function()
    # print("Loss function initialized successfully:", criterion)
    print("Loss function initialized successfully.")

    # Step 12: Get the device to be used for training
    device = get_device(config["training"]["device"])

    # Step 13: Initialize Trainer with all the components
    output_path = config["output"]["save_dir"]
    epochs = config["training"]["epochs"]
    early_stop = config["training"].get("early_stop")
    logging_config = config["training"].get("logging")

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
        device,
        output_path,
        epochs,
        scheduler=scheduler,
        early_stop=early_stop,
        logging_config=logging_config,
    )
    trainer.train()
    trainer.close()


if __name__ == "__main__":
    main()
