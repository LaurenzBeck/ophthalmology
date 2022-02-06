# commands

these are the commands to run the experiments on the jku server

## diabetic retinopathy desease grading - supervised baseline

1. Resnet18 42 epochs, strong aug, batch_size 42 focal loss gamma 2

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd model=resnet18 datamodule.batch_size=42 lightning_module/loss=focal logger.run_name=resnet18_focal_loss_strong_aug_42_epochs save_model="resnet18_focal_strong_aug.pt" trainer.gpus=[0]
```

2. Resnet18 42 epochs, strong aug, batch_size 42, image size 128

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd model=resnet18 image_size=128 datamodule.batch_size=42 logger.run_name=resnet18_128image_strong_aug_42_epochs save_model="resnet18_128image_strong_aug.pt" trainer.gpus=[1]
```

3. ImageNet pre-trained Resnet18 42 epochs, strong aug, batch_size 42

```bash
python ophthalmology/scripts/train_disease_grading.py environment=jku_ssd model=resnet18 model.pretrained=True datamodule.batch_size=42 logger.run_name=pretrained_resnet18_strong_aug_42_epochs save_model="pretrained_resnet18_strong_aug.pt" trainer.gpus=[2]
```

## diabetic retinopathycontrastive pre-training

4. Resnet50 24 epochs, simclr aug, batch_size 42 image size 128

```bash
python ophthalmology/scripts/train_simclr.py environment=jku_ssd datamodule.batch_size=42 image_size=128 trainer.max_epochs=24 logger.run_name=simclr_aug_24_epochs  save_model="resnet50backbone_128image_simclr_aug.pt" trainer.gpus=[3]
```
