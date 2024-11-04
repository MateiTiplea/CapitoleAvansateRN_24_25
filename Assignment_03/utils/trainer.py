import json
import os

import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        device,
        output_path,
        epochs,
        scheduler=None,
        early_stop=None,
        logging_config=None,
    ):
        """
        Initializes the Trainer with model, optimizer, criterion, data loaders, device, output path, and scheduler.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer to be used for training.
            criterion (nn.Module): The loss function to be used for training.
            train_loader (DataLoader): The DataLoader for training data.
            test_loader (DataLoader): The DataLoader for test data.
            device (torch.device): The device to be used for training.
            output_path (str): The path to save the trained model.
            epochs (int): The number of epochs to train the model.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            early_stop (dict): The early stopping configuration.
            logging_config (dict): The logging configuration.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_path = output_path
        self.epochs = epochs
        self.scheduler = scheduler
        self.scaler = GradScaler()

        if early_stop:
            self.early_stopping_patience = early_stop.get("patience", 5)
            self.early_stopping_delta = early_stop.get("delta", 0.0)
            self.monitor_metric = early_stop.get("monitor_metric", "best_test_loss")
            self.best_metric_value = (
                float("inf") if "loss" in self.monitor_metric else -float("inf")
            )
            self.early_stopping_counter = 0
            self.early_stop = False
        else:
            self.early_stopping_patience = None  # Indicates no early stopping

        self.model = self.model.to(self.device)
        self.model = torch.jit.script(self.model)

        self.best_metrics = {
            "best_train_accuracy": 0,
            "best_test_accuracy": 0,
            "best_loss": float("inf"),
            "best_test_loss": float("inf"),
        }

        os.makedirs(self.output_path, exist_ok=True)

        self.logging_config = logging_config or {}

        # TensorBoard setup
        self.tensorboard_writer = None
        if self.logging_config.get("tensorboard", False):
            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(output_path, "tensorboard_logs")
            )

        # wandb setup
        self.wandb_enabled = self.logging_config.get("wandb", {}).get("enabled", False)
        if self.wandb_enabled:
            wandb.init(
                project=self.logging_config["wandb"].get("project", "default_project"),
                entity=self.logging_config["wandb"].get("entity"),
                config=self.logging_config["wandb"].get("config", {}),
            )
            wandb.watch(self.model, log="all", log_freq=100)

    @staticmethod
    def accuracy(output: Tensor, labels: Tensor):
        fp_plus_fn = torch.logical_not(output == labels).sum().item()
        all_elements = len(output)
        return (all_elements - fp_plus_fn) / all_elements

    def _check_early_stopping(self, current_metric_value):
        """
        Checks if early stopping criteria are met and updates the patience counter.
        """
        if (
            self.early_stopping_patience is None
        ):  # Skip if early stopping is not enabled
            return

        # Determine if there is an improvement based on the delta and monitor metric
        if "loss" in self.monitor_metric:
            improved = (
                current_metric_value
                < self.best_metric_value - self.early_stopping_delta
            )
        else:
            improved = (
                current_metric_value
                > self.best_metric_value + self.early_stopping_delta
            )

        if improved:
            self.best_metric_value = current_metric_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stop = True

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0

        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type):
                output = self.model(inputs)
                loss = self.criterion(output, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()

            predicted = output.argmax(dim=1)

            # labels = labels.argmax(dim=1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            torch.cuda.empty_cache()

        acc = round(correct / total, 4)
        loss = round(running_loss / len(self.train_loader), 4)

        return acc, loss

    def evaluate(self):
        self.model.eval()

        all_outputs = []
        all_labels = []

        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                output = self.model(inputs)
                loss = self.criterion(output, labels)

                running_loss += loss.item()
                output = output.softmax(dim=1).cpu().squeeze()
                labels = labels.cpu().squeeze()

                all_outputs.append(output)
                all_labels.append(labels)

                torch.cuda.empty_cache()

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        acc = round(self.accuracy(all_outputs, all_labels), 4)
        loss = round(running_loss / len(self.test_loader), 4)

        return acc, loss

    def _do_epoch(self):
        acc, loss = self._train_epoch()
        acc_val, loss_val = self.evaluate()

        return acc, acc_val, loss, loss_val

    def _save_checkpoint(self, metric_name):
        """
        Saves a checkpoint for the current model state based on a specific metric.

        Args:
            metric_name (str): The metric name for which the checkpoint is being saved.
        """
        checkpoint_path = os.path.join(self.output_path, f"{metric_name}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def _update_best_metrics(self, acc, acc_val, loss, loss_val):
        metrics = {
            "best_train_accuracy": acc,
            "best_test_accuracy": acc_val,
            "best_loss": loss,
            "best_test_loss": loss_val,
        }

        for key, value in metrics.items():
            if key.startswith("best_"):
                if key == "best_train_accuracy":
                    if value > self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self._save_checkpoint(key)
                elif key == "best_test_accuracy":
                    if value > self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self._save_checkpoint(key)
                elif key == "best_loss":
                    if value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self._save_checkpoint(key)
                elif key == "best_test_loss":
                    if value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self._save_checkpoint(key)

    def _log_metrics(self, epoch, acc, acc_val, loss, loss_val):
        """
        Logs metrics to TensorBoard and wandb if enabled.
        """
        metrics = {
            "Train/Accuracy": acc,
            "Test/Accuracy": acc_val,
            "Train/Loss": loss,
            "Test/Loss": loss_val,
        }

        # Log to TensorBoard
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, epoch)

        # Log to wandb
        if self.wandb_enabled:
            wandb.log(metrics, step=epoch)

    def close(self):
        """Closes the TensorBoard writer and finalizes wandb."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        if self.wandb_enabled:
            wandb.finish()

    def train(self):
        self.epochs = tuple(range(self.epochs))
        current_epoch = 0
        with tqdm(self.epochs) as tbar:
            for _ in tbar:
                current_epoch += 1
                acc, acc_val, loss, loss_val = self._do_epoch()
                if self.scheduler:
                    self.scheduler.step(acc_val)

                self._update_best_metrics(acc, acc_val, loss, loss_val)

                if self.early_stopping_patience is not None:
                    self._check_early_stopping(
                        current_metric_value=self.best_metrics[self.monitor_metric]
                    )

                self._log_metrics(current_epoch, acc, acc_val, loss, loss_val)

                tbar.set_description(
                    f"Train Acc: {acc:.4f}, Test Acc: {acc_val:.4f}, Loss: {loss:.4f}, Test Loss: {loss_val:.4f}"
                )

                if self.early_stopping_patience is not None:
                    if self.early_stop:
                        print("Early stopping activated.")
                        break

        with open(os.path.join(self.output_path, "best_metrics.json"), "w") as f:
            json.dump(self.best_metrics, f, indent=4)
