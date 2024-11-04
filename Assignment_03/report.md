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
