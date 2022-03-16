# Datasets

Three datasets from retina scans fundus images were considered:
- Diabetic Retinopathy Detection - eyePACS
- Indian Diabetic Retinopathy Image Dataset - IDRiD
- MedMNIST v2 - RetinaMNIST

[TOC]

## Diabetic Retinopathy - Problem

> Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. It is estimated to affect over 93 million people.
    The US Center for Disease Control and Prevention estimates that 29.1 million people in the US have diabetes and the World Health Organization estimates that 347 million people have the disease worldwide. Diabetic Retinopathy (DR) is an eye disease associated with long-standing diabetes. Around 40% to 45% of Americans with diabetes have some stage of the disease. Progression to vision impairment can be slowed or averted if DR is detected in time, however this can be difficult as the disease often shows few symptoms until it is too late to provide effective treatment.
    Currently, detecting DR is a time-consuming and manual process that requires a trained clinician to examine and evaluate digital color fundus photographs of the retina. By the time human readers submit their reviews, often a day or two later, the delayed results lead to lost follow up, miscommunication, and delayed treatment.
    Clinicians can identify DR by the presence of lesions associated with the vascular abnormalities caused by the disease. While this approach is effective, its resource demands are high. The expertise and equipment required are often lacking in areas where the rate of diabetes in local populations is high and DR detection is most needed. As the number of individuals with diabetes continues to grow, the infrastructure needed to prevent blindness due to DR will become even more insufficient.
    The need for a comprehensive and automated method of DR screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With color fundus photography as input, the goal of this competition is to push an automated detection system to the limit of what is possible – ideally resulting in models with realistic clinical potential. The winning models will be open sourced to maximize the impact such a model can have on improving DR detection.

- introduction taken from: https://www.kaggle.com/c/diabetic-retinopathy-detection/overview

---

## Diabetic Retinopathy Detection - eyePACS

link: https://www.kaggle.com/c/diabetic-retinopathy-detection/data

dataset statistic | value
--- | ---
image size | width: 4752, height: 3168
number of training samples | 35126
number of testing samples | 53575

image sample:
![drd_sample](./images/diabetic_retinopathy_sample.jpeg)

```bibtex
@dataset{eyepacs,
author = {Misra, Rishabh},
year = {2018},
month = {06},
pages = {},
title = {News Category Dataset},
doi = {10.13140/RG.2.2.20331.18729}
}
```

train samples with "strong" augmentation settings:
![drd_train_samples](./images/drd_train_samples.png)

---

## Indian Diabetic Retinopathy Image Dataset - IDRiD

link: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

image sample:
![drd_sample](./images/idrd_sample.jpg)

```bibtex
@data{idrid,
    doi = {10.21227/H25W98},
    url = {https://dx.doi.org/10.21227/H25W98},
    author = {Porwal, Prasanna and Pachade, Samiksha and Kamble, Ravi and Kokare, Manesh and Deshmukh, Girish and Sahasrabuddhe, Vivek and Meriaudeau, Fabrice},
    publisher = {IEEE Dataport},
    title = {Indian Diabetic Retinopathy Image Dataset (IDRiD)},
    year = {2018}
}
```

train samples with "strong" augmentation settings:
![idrid_train_samples](./images/idrid_train_samples.png)

---

## MedMNIST v2 - RetinaMNIST

link: https://medmnist.com/

dataset statistic | value
--- | ---
image size | width: 28, height: 28
number of training samples | 1080
number of validation samples | 120
number of testing samples | 400

image sample:
![retina_mnist_sample](./images/retina_mnist_sample.png)

```bibtex
@article{medmnistv2,
    title={MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification},
    author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
    journal={arXiv preprint arXiv:2110.14795},
    year={2021}
}
```

train samples with "strong" augmentation settings:
![retina_mnist_train_samples](./images/retina_mnist_train_samples.png)
