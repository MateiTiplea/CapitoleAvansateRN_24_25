## General Overview of the Custom Pipeline

The custom pipeline works by specifying a couple properties that are essential in a configuration file that looks something like this:

```yaml
dataset:
  name: "CIFAR100"
  data_dir: "./data/cifar100"
  download: True
  initial_transform_script: "./transforms/initial_transforms.py"
  runtime_transform_script: "./transforms/runtime_transforms_basic.py"

dataloader:
  train:
    num_workers: 0
    batch_size: 100
    drop_last: True
    persistent_workers: False
    shuffle: True
    pin_memory: True

  test:
    num_workers: 0
    batch_size: 500
    drop_last: False
    persistent_workers: False
    shuffle: False
    pin_memory: False

training:
  # model_file: "./models/mlp.py"
  # model_class: "MLP"
  model: "resnet18"
  epochs: 50
  loss: "CrossEntropyLoss"
  device: "cuda"
  optimizer:
    name: "SGD" # or "Adam", "AdamW", "RMSprop"
    params:
      lr: 0.005
      momentum: 0.9
      weight_decay: 0.005
      nesterov: True
  scheduler:
    name: "ReduceLROnPlateau"
    mode: "min"
    factor: 0.1
    patience: 5

  early_stop: # Optional early stopping parameters
    patience: 5
    delta: 0.0001
    monitor_metric: "best_test_accuracy"

  logging:
    tensorboard: true

output:
  save_dir: "./output/cifar100"
```

The configuration file contains 4 main sections:

1. Dataset: contains properties such as the dataset to be used (can be given by name or by path), the directory where the data is stored, whether to download the data, the initial transform script, and the runtime transform script.

   - the dataset can be given through its name if it is something predefined and accepted by the config validator such as: MNIST, CIFAR10, CIFAR100; or the name can be specified in the following form: `<module>.<class>` where the module is the path to the module and the class is the name of the class that is the custom dataset.
   - the initial transform script and runtime transform script are python script given by their path that each contain a function that returns the transform to be applied to the data; that function is then dynamically imported and used in the pipeline.

   ```python
   def get_initial_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                mean=(0.491, 0.482, 0.446),
                std=(0.247, 0.243, 0.261),
            ),
        ]
    )
   ```

   ```python
   def get_runtime_transform():
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)]
    )
   ```

2. Dataloader: contains properties such as the number of workers, batch size, whether to drop the last batch, whether to use persistent workers, whether to shuffle the data, and whether to pin the memory, for both the training and testing dataloaders.

3. Training: contains the model to be used, the number of epochs, the loss function to be used, the device to be run on, the optimizer to be used, the scheduler to be used, and the early stopping parameters, if any, and the logging parameters.

   - the model can be given through its name if it is something predefined and accepted by the config validator such as: "resnet18", "PreActResNet18", "MLP", etc.; or it can be specified in two steps, the model file and the model class, where the model file is the path to the module and the model class is the name of the class that is the custom model.
   - the loss function has to be one of the predefined loss functions in PyTorch, such as: "CrossEntropyLoss", "MSELoss"
   - the optimizer can be one of the predefined optimizers in PyTorch, such as: "SGD", "Adam", "AdamW", "RMSprop", and each optimizer has its own parameters that can be specified.
   - the scheduler can be one of the predefined schedulers in PyTorch, such as: "StepLR", "ReduceLROnPlateau", and each scheduler has its own parameters that can be specified.
   - the early stopping parameters are optional and can be specified if early stopping is desired, and they contain the patience, delta, and monitor metric.
   - the logging parameters contain the tensorboard flag that specifies whether to log the training process using tensorboard, or even wandb.

4. Output: contains the directory where the output will be saved.

## Experimentation and Results

The parameter sweep was done using a combination of wandb and a custom script. The wandb was used to create multiple sweep files with combinations of the parameters. The custom script was used in order to get the sweep files, create appropiate configurations files for the custom pipeline, and then run those configurations. Each result is then saved in the "./output" directory with the name of the configuration file.

---

The parameters that were varied during the sweep were:

- model: between "resnet18", "PreActResNet18", and later 2 tests have been made with "vgg16"
- runtime_transform_script: "transforms/runtime_transform_basic.py", "transforms/runtime_transform_advanced.py", since the way the pipeline is implemented, it takes the script path as a parameter and then dynamically imports the script
- optimizer: "adam", "sgd"
- learning_rate: 0.001, 0.01, 0.005
- momentum: 0.9, 0.95
- weight_decay: 0.0001, 0.001

---

Some results are presented in the table bellow:

| Config          | best_train_accuracy | best_test_accuracy | best_loss | best_test_loss |
| --------------- | ------------------- | ------------------ | --------- | -------------- |
| config-1fyzcixl | 0.8462              | 0.6963             | 0.5397    | 1.0811         |
| config-1orzxxkf | 0.8944              | 0.6863             | 0.3529    | 1.1895         |
| config-25ve3s2t | 0.7302              | 0.5824             | 0.892     | 1.6542         |
| config-32dtkxxb | 0.7277              | 0.5816             | 0.9079    | 1.6455         |
| config-4j10zfd0 | 0.6271              | 0.572              | 1.2992    | 1.5391         |
| config-4jsilxod | 0.763               | 0.5887             | 0.7837    | 1.6023         |
| config-5qkqtu6k | 0.5622              | 0.4967             | 1.5582    | 1.9935         |
| config-6l6kvmej | 0.8469              | 0.6867             | 0.5083    | 1.1532         |
| config-7h4cxsim | 0.792               | 0.6694             | 0.7156    | 1.1718         |
| config-7hi2nbat | 0.6883              | 0.609              | 1.1037    | 1.3928         |
| config-vgg16-01 | 0.905               | 0.7009             | 0.3863    | 1.1021         |
| config-vgg16-02 | 0.8391              | 0.6751             | 0.5914    | 1.185          |

---

Considering the following aspects, I believe that the implemented pipeline is pretty efficient:

1. Device-Agnostic Design
   - GPU Utilization: The pipeline is designed to leverage GPUs effectively by handling data and model computations on the device using torch.cuda and torch.device. Operations like tensor transfers and computations are minimized by ensuring all data handling occurs on the GPU wherever possible.
   - Automatic Mixed Precision (AMP): The pipeline employs torch.cuda.amp for automatic mixed precision, reducing the memory footprint and increasing training speed by handling computations in half precision where appropriate. This allows the model to process larger batches per GPU memory unit, enhancing training efficiency.
2. Memory Efficiency and Gradient Scaling
   - Gradient Scaling and GradScaler: The pipeline employs GradScaler to dynamically adjust gradient values, preventing potential overflow issues associated with AMP. This ensures stable training and reduces memory usage by eliminating the need for additional full-precision copies of tensors, allowing larger batch sizes and increasing GPU utilization.
   - Efficient Backpropagation: By leveraging the mixed-precision scaling capabilities of PyTorch, the pipeline avoids recalculating unnecessary gradients, further optimizing the training loop and minimizing memory overhead.
3. Configurable Model and Training Parameters
   - Flexible Model and Optimizer Choices: Supporting different model architectures (e.g., ResNet, PreActResNet) and optimizers (e.g., Adam, SGD) ensures the pipeline can adapt to the best fit for various tasks. Users can specify models and optimizers that optimize computational efficiency for their data, choosing simpler models for lower computational needs or complex architectures for larger datasets.
   - Learning Rate Schedulers: The pipeline integrates learning rate schedulers like StepLR and ReduceLROnPlateau to adjust the learning rate dynamically. This avoids the risk of small updates slowing down convergence or large updates destabilizing training, leading to faster convergence times overall.
