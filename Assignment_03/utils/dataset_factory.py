import importlib.util
import os

from torch.utils.data import Dataset
from torchvision import datasets, transforms


def get_default_transforms(dataset_name):
    """Returns default transforms for the specified dataset."""
    if dataset_name == "MNIST":
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
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
    else:
        return None  # No default transforms for unsupported datasets


def load_transform_from_script(script_path, function_name):
    """
    Dynamically loads a transform function from a script.

    Args:
        script_path (str): Path to the script containing the transform function.
        function_name (str): Name of the function to retrieve from the script.

    Returns:
        Callable: The transform function, or None if not found.

    Raises:
        ValueError: If the script or function cannot be loaded.
    """
    if not os.path.isfile(script_path):
        raise ValueError(f"Transform script not found: {script_path}")

    # Load the module from the given script path
    spec = importlib.util.spec_from_file_location("transform_module", script_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Error loading transform script '{script_path}': {e}")

    # Retrieve the specified function from the module
    if not hasattr(module, function_name):
        raise ValueError(
            f"Function '{function_name}' not found in script '{script_path}'"
        )

    return getattr(module, function_name)()


class CachedDataset(Dataset):
    def __init__(self, base_dataset, transform=None, runtime_transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.runtime_transform = runtime_transform
        self.cache = []
        self._cache_data()

    def _cache_data(self):
        print("Caching dataset in memory...")
        for i in range(len(self.base_dataset)):
            image, label = self.base_dataset[i]
            if self.transform:
                image = self.transform(image)
            self.cache.append((image, label))
        print("Dataset successfully cached.")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        image, label = self.cache[idx]
        if self.runtime_transform:
            image = self.runtime_transform(image)
        return image, label


class DatasetFactory:
    DEFAULT_DATASETS = {
        "MNIST": datasets.MNIST,
        "CIFAR10": datasets.CIFAR10,
        "CIFAR100": datasets.CIFAR100,
    }

    def __init__(
        self,
        dataset_name,
        data_dir,
        download=False,
        initial_transform_script=None,
        runtime_transform_script=None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.download = download

        # Load initial and runtime transforms from script paths if provided
        self.initial_transform = (
            load_transform_from_script(
                initial_transform_script, "get_initial_transform"
            )
            if initial_transform_script
            else get_default_transforms(dataset_name)
        )
        self.runtime_transform = (
            load_transform_from_script(
                runtime_transform_script, "get_runtime_transform"
            )
            if runtime_transform_script
            else None
        )

    def get_dataset(self, train=True):
        if self.dataset_name in self.DEFAULT_DATASETS:
            base_dataset = self._load_default_dataset(train)
        else:
            base_dataset = self._load_custom_dataset(train)

        return CachedDataset(
            base_dataset,
            transform=self.initial_transform,
            runtime_transform=self.runtime_transform,
        )

    def _load_default_dataset(self, train):
        try:
            dataset_class = self.DEFAULT_DATASETS[self.dataset_name]
            return dataset_class(
                root=self.data_dir, train=train, download=self.download
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load default dataset '{self.dataset_name}': {e}"
            )

    def _load_custom_dataset(self, train):
        if "." not in self.dataset_name:
            raise ValueError(
                "Custom dataset name must be in '<module_name>.<dataset_class>' format."
            )

        module_name, class_name = self.dataset_name.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            CustomDatasetClass = getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Could not import module '{module_name}' for custom dataset '{self.dataset_name}': {e}"
            )
        except AttributeError as e:
            raise ImportError(
                f"The specified class '{class_name}' was not found in module '{module_name}': {e}"
            )

        try:
            return CustomDatasetClass(root=self.data_dir, train=train)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate custom dataset '{self.dataset_name}': {e}"
            )
