import os

import torch
from torch import Tensor
from tqdm import tqdm


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

        self.model = self.model.to(self.device)
        self.model = torch.jit.script(self.model)

        self.best_metrics = {
            "best_train_accuracy": 0,
            "best_test_accuracy": 0,
            "best_fp_rate": float("inf"),
            "best_fn_rate": float("inf"),
            "best_loss": float("inf"),
            "best_test_loss": float("inf"),
        }

        os.makedirs(self.output_path, exist_ok=True)

    @staticmethod
    def accuracy(output: Tensor, labels: Tensor):
        fp_plus_fn = torch.logical_not(output == labels).sum().item()
        all_elements = len(output)
        return (all_elements - fp_plus_fn) / all_elements

    @staticmethod
    def false_positive_rate(output: Tensor, labels: Tensor):
        return (
            torch.logical_and(output == 1, labels == 0).sum().item()
            / (labels == 0).sum().item()
        )

    @staticmethod
    def false_negative_rate(output: Tensor, labels: Tensor):
        return (
            torch.logical_and(output == 0, labels == 1).sum().item()
            / (labels == 1).sum().item()
        )

    def _train_epoch(self):
        self.model.train()

        all_outputs = []
        all_labels = []

        running_loss = 0.0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            output = self.model(inputs)
            loss = self.criterion(output, labels)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            output = output.softmax(dim=1).detach().cpu().squeeze()
            labels = labels.cpu().squeeze()

            all_outputs.append(output)
            all_labels.append(labels)

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        acc = round(self.accuracy(all_outputs, all_labels), 4)
        loss = round(running_loss / len(self.train_loader), 4)

        return acc, loss

    def evaluate(self):
        self.model.eval()

        all_outputs = []
        all_labels = []

        running_loss = 0.0

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

        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        all_labels = torch.cat(all_labels)

        acc = round(self.accuracy(all_outputs, all_labels), 4)
        loss = round(running_loss / len(self.test_loader), 4)
        fp_rate = round(self.false_positive_rate(all_outputs, all_labels), 4)
        fn_rate = round(self.false_negative_rate(all_outputs, all_labels), 4)

        return acc, loss, fp_rate, fn_rate

    def _do_epoch(self):
        acc, loss = self._train_epoch()
        acc_val, loss_val, fp_rate, fn_rate = self.evaluate()

        return acc, acc_val, fp_rate, fn_rate, loss, loss_val

    def _save_checkpoint(self, metric_name):
        """
        Saves a checkpoint for the current model state based on a specific metric.

        Args:
            metric_name (str): The metric name for which the checkpoint is being saved.
        """
        checkpoint_path = os.path.join(self.output_path, f"{metric_name}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def _update_best_metrics(self, acc, acc_val, fp_rate, fn_rate, loss, loss_val):
        metrics = {
            "best_train_accuracy": acc,
            "best_test_accuracy": acc_val,
            "best_fp_rate": fp_rate,
            "best_fn_rate": fn_rate,
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
                elif key == "best_fp_rate":
                    if value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                        self._save_checkpoint(key)
                elif key == "best_fn_rate":
                    if value < self.best_metrics[key]:
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

    def train(self):
        self.epochs = tuple(range(self.epochs))
        current_epoch = 0
        with tqdm(self.epochs) as tbar:
            for _ in tbar:
                current_epoch += 1
                acc, acc_val, fp_rate, fn_rate, loss, loss_val = self._do_epoch()
                if self.scheduler:
                    self.scheduler.step(acc_val)

                self._update_best_metrics(
                    acc, acc_val, fp_rate, fn_rate, loss, loss_val
                )
                tbar.set_description(
                    f"Train Acc: {acc:.4f}, Test Acc: {acc_val:.4f}, FP Rate: {fp_rate:.4f}, FN Rate: {fn_rate:.4f}, Loss: {loss:.4f}, Test Loss: {loss_val:.4f}"
                )
