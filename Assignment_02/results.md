| Experiments                                                                           | Best Val Accuracy |
| ------------------------------------------------------------------------------------- | ----------------- |
| base                                                                                  | 40.25             |
| 1. Adam                                                                               | 53.05             |
| 2. Normalization, SGD, lr 0.001, momentum 0.9, nesterov, L2 0.005                     | 61.89             |
| 3. Normalization, SGD, lr 0.001, momentum 0.9, nesterov, L2 0.005, LRS                | 61.40             |
| 4. Normalization, SGD, lr 0.001, momentum 0.9, nesterov, L2 0.005, LRS, batch_size 50 | 61.40             |
| 5. DA-v1, SGD, lr 0.001, momentum 0.9, nesterov, L2 0.005, LRS, batch_size 50         | 62.19             |
| 6. DA-v2, SGD, lr 0.001, momentum 0.9, nesterov, L2 0.005, LRS, batch_size 50         | 65.30             |
| 7. DA-v3, SGD, lr 0.001, momentum 0.9, nesterov, L2 0.005, LRS, batch_size 50         | 69.35             |

1. Uses Adam optimizer - [test_1.py](test_1.py)
2. Uses Normalization (mean, std), SGD optimizer, lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.005 - [test_02.py](test_02.py)
3. - Uses Normalization
   - Uses SGD optimizer with lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.005
   - Uses Learning Rate Scheduler - ReduceLROnPlateau with patience=10, factor=0.2, threshold=0.001, threshold_mode='rel'
   - [test_03.py](test_03.py)
4. - Uses Normalization
   - Uses SGD optimizer with lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.005
   - Uses Learning Rate Scheduler - ReduceLROnPlateau with patience=10, factor=0.2, threshold=0.001, threshold_mode='rel'
   - Uses batch_size=50
   - [test_04.py](test_04.py)
5. - Uses Data Augmentation: RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomErasing
   - Uses Normalization
   - Uses SGD optimizer with lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.005
   - Uses Learning Rate Scheduler - ReduceLROnPlateau with patience=10, factor=0.2, threshold=0.001, threshold_mode='rel'
   - Uses batch_size=50
   - [test_05.py](test_05.py)
6. - Uses Data Augmentation: RandomResizedCrop, RandomAffine
   - Uses Normalization
   - Uses SGD optimizer with lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.005
   - Uses Learning Rate Scheduler - ReduceLROnPlateau with patience=5, factor=0.1, threshold=0.001, threshold_mode='rel'
   - Uses batch_size=50
   - [test_06.py](test_06.py)
7. - Uses Data Augmentation: RandomCrop, RandomAffine with Rotation and Translation
   - Uses Normalization
   - Uses SGD optimizer with lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.005
   - Uses Learning Rate Scheduler - ReduceLROnPlateau with patience=5, factor=0.1, threshold=0.001, threshold_mode='rel'
   - Uses batch_size=50
   - [test_07.py](test_07.py)
