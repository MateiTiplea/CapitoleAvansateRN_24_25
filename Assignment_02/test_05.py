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
                v2.RandomVerticalFlip(),
                v2.RandomErasing(),
            ]
        )
    return CachedDataset(cifar100, runtime_transforms, True)


# basic_transforms = v2.Compose(
#     [
#         v2.ToImage(),
#         v2.ToDtype(torch.float32, scale=True),
#         v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True),
#     ]
# )

# train_set = CIFAR100(
#     r"kaggle\input\fii-atnn-2024-assignment-2",
#     download=False,
#     train=True,
#     transform=basic_transforms,
# )
# test_set = CIFAR100(
#     r"kaggle\input\fii-atnn-2024-assignment-2",
#     download=False,
#     train=False,
#     transform=basic_transforms,
# )
# train_set = SimpleCachedDataset(train_set)
# test_set = SimpleCachedDataset(test_set)

# train_loader = DataLoader(train_set, batch_size=50, shuffle=True, pin_memory=pin_memory)
# test_loader = DataLoader(test_set, batch_size=500, pin_memory=pin_memory)


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


learning_rate = 0.001
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
    factor=0.2,
    patience=10,
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
    batch_size=50,
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


def train():
    model.train()
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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


best = 0.0
epochs = list(range(50))
with tqdm(epochs) as tbar:
    for epoch in tbar:
        train_acc = train()
        val_acc = val()

        scheduler.step(val_acc)
        if val_acc > best:
            best = val_acc
        tbar.set_description(
            f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}"
        )


data = {"ID": [], "target": []}


for i, label in enumerate(inference()):
    data["ID"].append(i)
    data["target"].append(label)

df = pd.DataFrame(data)
# df.to_csv("/kaggle/working/submission.csv", index=False)
df.to_csv(r"kaggle\working\submission.csv", index=False)
