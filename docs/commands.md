# commands

these are the commands to run the experiments on the jku server

## diabetic retinopathy desease grading - supervised baseline

1. ImageNet pre-trained Resnet18 42 epochs, strong aug, batch_size 42 focal loss gamma 2 balanced loader

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd model=resnet18 datamodule.batch_size=42 lightning_module/loss=focal +datamodule.balanced_sampling=True logger.run_name=balanced_resnet18_focal_loss_strong_aug_42_epochs save_model="pretrained_resnet18_focal_strong_aug_balanced.pt" trainer.gpus=[1]
```

2. ImageNet pre-trained Resnet18 42 epochs, strong aug, batch_size 42

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd model=resnet18 datamodule.batch_size=42 lightning_module/loss=weighted_cross_entropy logger.run_name=resnet18_strong_aug_42_epochs_weighted save_model="pretrained_resnet18_strong_aug_weighted.pt" trainer.gpus=[2]
```

4. ImageNet pre-trained Resnet18 42 epochs, strong aug, batch_size 42 regression

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd model=resnet18 model.num_output_units=1 datamodule.batch_size=42 lightning_module=disease_grading_regression lightning_module/loss=mse logger.run_name=resnet18_regression_strong_aug_42_epochs save_model="pretrained_resnet18_regression_strong_aug.pt" trainer.gpus=[1]
```

## diabetic retinopathycontrastive pre-training

4. ImageNet pre-trained Resnet50 42 epochs, simclr aug, batch_size 42 image size 256, balanced loader

```bash
python ophthalmology/scripts/train_simclr.py environment=jku_ssd datamodule.batch_size=42 trainer.max_epochs=42 transforms@ssl_transforms=strong_normalize logger.run_name=balanced_pretrained_strong_aug_42_epochs save_model="pretrained_resnet50backbone_256image_balanced_strong_aug.pt" trainer.gpus=[0]
```
