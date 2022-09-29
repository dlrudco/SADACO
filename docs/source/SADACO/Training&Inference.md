---
sort: 3
---

# Training and Inference

## Training

With sadaco installed, we can now start training on prepared datasets. Here, we provide example pipeline on ICBHI dataset.

First, import sadaco library, and build a trainer instance that will help us train a deep model.

In order to build a trainer, we have to first load a trainer configuration written in `yaml` format.
<details>
  <summary>YAML template example</summary>
  <b> Master Config </b>

  ```yaml
# TODO: We are planning to modify this structure with inheritance feature enabled, as in the detectron2(https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml)
prefix : <PREFIX>
project_name : <PROJECT NAME> #This will be used as a wandb project name
use_wandb : !!bool True
data_configs :
    file : <DATA_CONFIG_FILE_PATH>
    split : [train, test]
model_configs:
    file : <MODEL_CONFIG_FILE_PATH>
    resume : False # If True, model will resume to the .pth file specified in MODEL_CONFIG
    resume_optimizer : False # If True, optimizer will also resume. Else, only model resumes.

output_dir : checkpoints/

train: # Training Pipeline Configuration
    method : basic # TODO: Currently not handled by the trainer. 
    target_metric : F1-Score # Target metric that will be used to determine the best model.
    max_epochs : !!int 650
    save_interval : !!int 1 # Model saving interval. Only saves the last if -1.
    update_interval : !!int 6 # Gradient Accumulation interval.
    criterion:
        name: CELoss
        loss_mixup : !!bool True
        params : 
            mode : onehot
            reduction : mean
    optimizer:
        name: Adam
        params:
        lr : !!float 3e-6
        weight_decay : !!float 5e-7
        betas : !!python/tuple [0.95, 0.999]
    lr_scheduler:
        name : CosineAnnealingWarmUpRestarts
        params :
        T_0: !!int 40
        T_mult : !!int 1
        eta_max : !!float 5e-4
        T_up : !!int 10
        gamma: !!float 1.


data:
    train_dataloader:
        sampler : 
            name : BalancedBatchSampler
            params:
                n_classes : 4
                n_samples : 10
        params : 
            shuffle : True
            batch_size : 128
            num_workers : 8
            pin_memory : True
            persistent_workers : True
            drop_last : False
    val_dataloader:
        params : 
            shuffle : False
            batch_size : 16
            num_workers : 8
            pin_memory : True
            persistent_workers : False
            drop_last : False
  

  ```
</details>



```python
import sadaco
from sadaco.utils import config_parser

master_config_path = '<MASTER_CONFIG_PATH>'
my_configs = config_parser(master_config_path)
```