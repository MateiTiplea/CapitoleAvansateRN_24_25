from torchvision import transforms


def get_initial_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                mean=(0.491, 0.482, 0.446),
                std=(0.247, 0.243, 0.261),
            ),
        ]
    )
