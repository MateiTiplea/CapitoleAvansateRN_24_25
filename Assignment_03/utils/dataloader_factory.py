import torch
from torch.utils.data import DataLoader


class DataLoaderFactory:
    def __init__(self, dataloader_config):
        """
        Initializes the DataLoaderFactory with configuration.

        Args:
            dataloader_config (dict): Configuration dictionary for DataLoader settings,
                                      containing 'train' and 'test' sections.
        """
        self.dataloader_config = dataloader_config

    def create_dataloader(self, dataset, mode="train"):
        """
        Creates and returns a DataLoader for the specified dataset and mode.

        Args:
            dataset (Dataset): The dataset object to be loaded.
            mode (str): Mode for which DataLoader is being created; should be either 'train' or 'test'.

        Returns:
            DataLoader: A DataLoader configured according to the settings in the configuration.

        Raises:
            ValueError: If an invalid mode is specified.
        """
        if mode not in self.dataloader_config:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'train' or 'test'.")

        # Extract configuration for the specified mode
        config = self.dataloader_config[mode]

        # Create DataLoader with the specified settings
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=config["num_workers"],
            drop_last=config["drop_last"],
            pin_memory=config["pin_memory"],
            persistent_workers=config["persistent_workers"],
        )
