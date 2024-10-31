from torchvision import transforms


def get_runtime_transform():
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    )
