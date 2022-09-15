![image_header](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fa1anqn8ned-flywheel.netdna-ssl.com%2Fwp-content%2Fuploads%2F2019%2F03%2Foph-post_03222019.jpg&f=1&nofb=1)

<h1 align="center">👁 - Contrastive Learning for Ophthalmology - 👁</h1>

<p align="center">
    Seminar in AI - JKU Linz
</p>

<p align="center">
    <a href="https://www.repostatus.org/#inactive"><img src="https://www.repostatus.org/badges/latest/inactive.svg" alt="Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows." /></a>
</p>

<p align="center">
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.4.9+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
    <a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
</p>

<p align="center">
  links to the official docs:
</p>
<p align="center">
  <a href="https://laurenzbeck.github.io/ophthalmology/docs/datasets/">💾 Datasets</a> •
  <a href="https://laurenzbeck.github.io/ophthalmology/docs/experiments/">🔬 Experiments</a> •
  <a href="https://laurenzbeck.github.io/ophthalmology/reference/ophthalmology/">🐍 API Reference</a>
</p>

---

## Project

This project was part of my master studies in Artificial Intelligence at the Johannes Kepler University in Linz.
The goal of the Seminar "practical work in AI" was to conduct proper research and experiments in a chosen field.

I decided to join the Machine Learning Institute for Life Science and was supervised by my two professors:

+ Andreas Fürst
+ Elisabeth Rumetshofer

The domain of the project is computer vision applied to high-resolution retina scans to help developing models that can
support physicians diagnosing certain eye deseases.

## Task

Following current research trends from 2020/2021, contrastive representation learning methods and the influence of the view
selection and generation process on the downstream performance was chosen as the main research focus.

For a proper downstream evaluation of the learned representations, two tasks on retina scans are used:

1. deasease grading of diabetic retinopathy
2. localization of important landmarks of the retina

## Dataset

![retina scan](docs/images/diabetic_retinopathy_sample.jpeg)

Three public datasets were used during the experiments of the project:

1. [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
2. [Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
3. [Retina MNIST](https://medmnist.com/)

## Installation

To install the projects dependencies and create a virtual environment, make sure that your system has python (>=3.9,<3.10) and [poetry](https://python-poetry.org/) installed.

Then `cd` into the projects root directory and call: `$poetry install`
