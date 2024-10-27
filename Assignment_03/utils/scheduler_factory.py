# utils/scheduler_factory.py
import torch.optim.lr_scheduler as lr_scheduler


class SchedulerFactory:
    def __init__(self, config):
        """
        Initializes the SchedulerFactory with configuration.

        Args:
            config (dict): Configuration dictionary containing scheduler specifications.
        """
        self.config = config["training"].get("scheduler", None)

    def get_scheduler(self, optimizer):
        """
        Returns an instance of the specified scheduler or None if no scheduler is configured.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is being created.

        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: The specified scheduler instance, or None if not specified.
        """
        if not self.config:
            print(
                "No scheduler specified. Proceeding without a learning rate scheduler."
            )
            return None

        scheduler_name = self.config.get("name")
        params = self.config.get("params", {})

        if scheduler_name == "StepLR":
            return lr_scheduler.StepLR(optimizer, **params)
        elif scheduler_name == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        else:
            raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")
