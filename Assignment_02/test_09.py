import os
import random
from typing import Optional

import pandas as pd
import torch
from torch import GradScaler, Tensor, nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import v2
from tqdm import tqdm

device = torch.device("cuda")
cudnn.benchmark = True
pin_memory = True
enable_half = True  # Disable for CPU, it is slower!
scaler = GradScaler(device, enabled=enable_half)


class CachedDataset(Dataset):
    def __init__(
        self, dataset: Dataset, runtime_transforms: Optional[v2.Transform], cache: bool
    ):
        if cache:
            dataset = tuple([x for x in dataset])
        self.dataset = dataset
        self.runtime_transforms = runtime_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        if self.runtime_transforms is None:
            return image, label
        return self.runtime_transforms(image), label


def get_dataset(data_path: str, is_train: bool):
    initial_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261)),
        ]
    )
    cifar100 = CIFAR100(
        root=data_path, train=is_train, transform=initial_transforms, download=True
    )
    runtime_transforms = None
    if is_train:
        runtime_transforms = v2.Compose(
            [
                v2.RandomCrop(size=32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.RandomAffine(
                    degrees=10, translate=(0.05, 0.05)
                ),  # Less aggressive affine
            ]
        )

    return CachedDataset(cifar100, runtime_transforms, True)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


learning_rate = 0.005
momentum = 0.9
weight_decay = 0.005
model = VGG16().to(device)
model = torch.jit.script(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
    nesterov=True,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.1,
    patience=5,
    threshold=0.001,
    threshold_mode="rel",
)

train_dataset = get_dataset(r"kaggle\input\fii-atnn-2024-assignment-2", is_train=True)
val_dataset = get_dataset(r"kaggle\input\fii-atnn-2024-assignment-2", is_train=False)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    pin_memory=pin_memory,
    num_workers=0,
    batch_size=100,
    drop_last=True,
)

test_loader = DataLoader(
    val_dataset,
    shuffle=False,
    pin_memory=True,
    num_workers=0,
    batch_size=500,
    drop_last=False,
)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()

    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    bbx1 = torch.clip(cx - cut_w // 2, 0, W).int().item()
    bby1 = torch.clip(cy - cut_h // 2, 0, H).int().item()
    bbx2 = torch.clip(cx + cut_w // 2, 0, W).int().item()
    bby2 = torch.clip(cy + cut_h // 2, 0, H).int().item()

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    lam = torch.clamp(lam, 0.3, 0.7)
    rand_index = torch.randperm(x.size()[0])
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def mixup_data(x, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    lam = torch.clamp(lam, 0.3, 0.7)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    y_a, y_b = y, y[rand_index]
    return mixed_x, y_a, y_b, lam


def criterion_mixup(pred, y_a, y_b, lam):
    return lam * nn.functional.cross_entropy(pred, y_a) + (
        1 - lam
    ) * nn.functional.cross_entropy(pred, y_b)


def train(cutmix_prob=0.2, mixup_prob=0.2):
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        r = random.random()
        if r < cutmix_prob:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
        elif r < cutmix_prob + mixup_prob:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
        else:
            targets_a, targets_b, lam = targets, targets, 1.0

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion_mixup(outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


@torch.inference_mode()
def val():
    model.eval()
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


@torch.inference_mode()
def val_with_softmax():
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        # Apply softmax to convert logits to probabilities
        softmax_outputs = torch.softmax(outputs, dim=1)

        # Use the highest probability for the final prediction
        predicted = softmax_outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Return the accuracy
    return 100.0 * correct / total


@torch.inference_mode()
def val_with_tta():
    model.eval()
    correct = 0
    total = 0

    # Define the test-time augmentations
    augmentations = [
        lambda x: x,  # original image
        v2.functional.hflip,  # horizontal flip
        lambda x: v2.functional.affine(
            x, angle=5, translate=(0, 0), scale=1.0, shear=0
        ),  # slight rotation
        lambda x: v2.functional.affine(
            x, angle=-5, translate=(0, 0), scale=1.0, shear=0
        ),  # slight rotation in the other direction
    ]

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )

        avg_logits = None  # Initialize to accumulate logits
        with torch.autocast(device.type, enabled=enable_half):
            for aug in augmentations:
                augmented_inputs = aug(inputs)
                logits = model(augmented_inputs)

                # Accumulate logits
                if avg_logits is None:
                    avg_logits = logits
                else:
                    avg_logits += logits

        avg_logits /= len(augmentations)  # Average the logits

        # Now apply softmax after averaging logits
        softmax_outputs = torch.softmax(avg_logits, dim=1)

        # Use the highest probability for the final prediction
        predicted = softmax_outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Return the accuracy
    return 100.0 * correct / total


@torch.inference_mode()
def inference():
    model.eval()

    labels = []

    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)

        predicted = outputs.argmax(1).tolist()
        labels.extend(predicted)

    return labels


if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

best = 0.0
epochs = list(range(100))
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        val_acc = val()

        # val_with_softmax_acc = val_with_softmax()
        # val_with_tta_acc = val_with_tta()

        scheduler.step(val_acc)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), os.path.join("checkpoints", "best.pth"))
        # tbar.set_description(
        #     f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}, Val with Softmax: {val_with_softmax_acc:.2f}, Val with TTA: {val_with_tta_acc:.2f}"
        # )
        tbar.set_description(
            f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}"
        )


model.load_state_dict(
    torch.load(
        os.path.join("checkpoints", "best.pth"),
        map_location=device,
        weights_only=True,
    )
)

best_val_acc = val()
print(f"Best validation accuracy: {best_val_acc:.2f}")

# best_val_acc_tta = val_with_tta()
# print(f"Best validation accuracy with TTA: {best_val_acc_tta:.2f}")

# best_val_acc_softmax = val_with_softmax()
# print(f"Best validation accuracy with Softmax: {best_val_acc_softmax:.2f}")

data = {"ID": [], "target": []}


for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
# df.to_csv("/kaggle/working/submission.csv", index=False)
df.to_csv(r"kaggle\working\submission.csv", index=False)
