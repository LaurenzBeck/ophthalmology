# Experiments

[TOC]

---

## Supervised Baselines

### sup_drd_grading
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=sup_drd_grading trainer.gpus=[1]
```

### sup_idrd_grading
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=sup_idrd_grading trainer.gpus=[3]
```

### sup_idrd_localization
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=sup_idrd_localization trainer.gpus=[0]
```

### sup_mnist_grading
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=sup_mnist_grading trainer.gpus=[2]
```

## Contrastive Pre-Training

## simclr_drd
```bash
python ophthalmology/scripts/pre_train.py environment=jku_ssd experiment=simclr_drd trainer.gpus=[2]
```


## Supervised Fine-tuning

### ft_drd_grading
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=ft_drd_grading trainer.gpus=[1]
```

### ft_idrd_grading
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=ft_idrd_grading trainer.gpus=[0]
```

### ft_idrd_localization
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=ft_idrd_localization trainer.gpus=[3]
```

### ft_mnist_grading
```bash
python ophthalmology/scripts/train.py environment=jku_ssd experiment=ft_mnist_grading trainer.gpus=[2]
```
