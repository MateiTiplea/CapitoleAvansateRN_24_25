import time

import numpy as np
import torch
from torchvision.datasets import MNIST
from tqdm import tqdm


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def initialize_weights(input_size, hidden_size, output_size):
    w1 = torch.randn(input_size, hidden_size) * 0.01
    b1 = torch.zeros(hidden_size)
    w2 = torch.randn(hidden_size, output_size) * 0.01
    b2 = torch.zeros(output_size)
    return w1, b1, w2, b2


def forward(x, w1, b1, w2, b2):
    z1 = x @ w1 + b1
    a1 = torch.relu(z1)  # First layer activation (ReLU)
    z2 = a1 @ w2 + b2
    a2 = torch.softmax(z2, dim=1)  # Output activation (Softmax)
    return z1, a1, z2, a2


def backward(x, y, z1, a1, z2, a2, w1, b1, w2, b2, lr):
    batch_size = y.size(0)

    # Output layer error
    error_output = a2 - y
    delta_w2 = a1.T @ error_output / batch_size
    delta_b2 = error_output.mean(dim=0)

    # Backpropagation to the hidden layer
    error_hidden = (error_output @ w2.T) * (z1 > 0)  # Derivative of ReLU
    delta_w1 = x.T @ error_hidden / batch_size
    delta_b1 = error_hidden.mean(dim=0)

    # Gradient descent step
    w1 = w1 - lr * delta_w1
    b1 = b1 - lr * delta_b1
    w2 = w2 - lr * delta_w2
    b2 = b2 - lr * delta_b2

    return w1, b1, w2, b2


def to_one_hot(labels, num_classes=10, device="cpu"):
    return torch.eye(num_classes, device=device)[labels]


def accuracy(predictions, labels):
    predicted_labels = torch.argmax(predictions, dim=1)
    return (predicted_labels == labels).float().mean().item()


def create_batches(data, batch_size):
    x_data, y_data = zip(*data)
    x_data = torch.stack(x_data)
    y_data = torch.stack(y_data)

    batches = []
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i : i + batch_size]
        y_batch = y_data[i : i + batch_size]
        batches.append((x_batch, y_batch))

    return batches


def train_epoch(data_loader, w1, b1, w2, b2, lr, device):
    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        x = x.view(-1, 28 * 28)
        y = to_one_hot(y, device=device)

        z1, a1, z2, a2 = forward(x, w1, b1, w2, b2)
        w1, b1, w2, b2 = backward(x, y, z1, a1, z2, a2, w1, b1, w2, b2, lr)

    return w1, b1, w2, b2


def evaluate(data_loader, w1, b1, w2, b2, device):
    total_acc = 0
    count = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(-1, 28 * 28)  # Flatten the images
        _, _, _, a2 = forward(x, w1, b1, w2, b2)
        total_acc += accuracy(a2, y) * x.size(0)
        count += x.size(0)

    return total_acc / count


def pil_to_tensor(image):
    image_array = np.array(image, dtype=np.float32)
    image_tensor = torch.from_numpy(image_array)
    return image_tensor / 255.0


def load_mnist():
    mnist_train = MNIST(root="data", train=True, download=True)
    mnist_test = MNIST(root="data", train=False, download=True)

    x_train = torch.stack([pil_to_tensor(img) for img, _ in mnist_train])
    x_test = torch.stack([pil_to_tensor(img) for img, _ in mnist_test])

    y_train = torch.tensor([y for _, y in mnist_train])
    y_test = torch.tensor([y for _, y in mnist_test])

    train_loader = list(zip(x_train, y_train))
    test_loader = list(zip(x_test, y_test))

    return train_loader, test_loader


def train(epochs, device):
    input_size = 28 * 28
    hidden_size = 100
    output_size = 10
    learning_rate = 0.1
    batch_size = 64

    w1, b1, w2, b2 = initialize_weights(input_size, hidden_size, output_size)

    train_loader, test_loader = load_mnist()

    w1, b1, w2, b2 = w1.to(device), b1.to(device), w2.to(device), b2.to(device)

    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_batches = create_batches(train_loader, batch_size)

        w1, b1, w2, b2 = train_epoch(
            train_batches, w1, b1, w2, b2, learning_rate, device
        )

        test_batches = create_batches(test_loader, batch_size)
        val_acc = evaluate(test_batches, w1, b1, w2, b2, device)

        print(f"Validation Accuracy: {val_acc*100:.2f}%")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds on {device}.")


if __name__ == "__main__":
    device = get_default_device()
    train(epochs=20, device=device)

    device = torch.device("cpu")
    train(epochs=20, device=device)
