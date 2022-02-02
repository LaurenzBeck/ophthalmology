# commands

these are the commands to run the experiments on the jku server

## diabetic retinopathy desease grading - supervised baseline

1. Resnet50 32 epochs, weak aug, batch_size 50

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd datamodule.batch_size=50 datamodule.num_workers=24 logger.experiment_name=disease_grading logger.run_name=resnet50_weak_aug_32_epochs save_model="resnet50_weak_aug.pt" trainer.gpus="0"
```

2. Resnet50 32 epochs, strong aug, batch_size 42

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd datamodule.batch_size=42 datamodule.num_workers=24 transforms@train_transforms=strong logger.experiment_name=disease_grading logger.run_name=resnet50_strong_aug_32_epochs save_model="resnet50_strong_aug.pt" trainer.gpus=[1]
```

3. Resnet18 18 epochs, weak aug, batch_size 42

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd datamodule.batch_size=42 datamodule.num_workers=24 model.name=resnet18 +model.num_resnet_features=512 logger.experiment_name=disease_grading logger.run_name=resnet18_weak_aug_32_epochs save_model="resnet18_weak_aug.pt" trainer.gpus=[2]
```

4. imagenet pretrained

## diabetic retinopathycontrastive pre-training

1. Resnet50 16 epochs, simclr aug, batch_size 36

```bash
python ophthalmology/scripts/train_simclr.py environment=jku_ssd datamodule.batch_size=36 datamodule.num_workers=24 trainer.max_epochs=16 logger.experiment_name=simclr logger.run_name=simclr_aug_16_epochs  save_model="resnet50backbone_simclr_aug.pt" trainer.gpus=[3]
```
