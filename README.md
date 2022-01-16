![image_header](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fa1anqn8ned-flywheel.netdna-ssl.com%2Fwp-content%2Fuploads%2F2019%2F03%2Foph-post_03222019.jpg&f=1&nofb=1)

<h1 align="center">👁 - Contrastive Learning for Ophthalmology - 👁</h1>

<p align="center">
    Seminar in AI - JKU Linz
</p>

<p align="center">
    <a href="https://www.repostatus.org/#wip"><img src="https://www.repostatus.org/badges/latest/wip.svg" alt="Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public." />
    </a>
</p>

> ## ⚠️ Active Development
> This Repository is in a very early concept/development stage and may drastically change. Use at your own risk.

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

The domain of the project is computer vison applied to high-resolution retina scans to help developing models that can
support physicians diagnosing certain eye deseases.

## Task

Following current research trends from 2020/2021, contrastive representation learning methods and the influence of the view
selection and generation process on the downstream performance was chosen as the main research focus.

For a proper downstream evaluation of the learned representations, three tasks on retina scans are used:

1. deasease grading of diabetic retinopathy
2. image segmentation of certain anatomic and pathogen regions
3. localization of important landmarks of the retina

## Dataset

![retina scan](docs/images/diabetic_retinopathy_sample.jpeg)

Three public datasets were used during the experiments of the project:

1. [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
2. [Indian Diabetic Retinopathy Image Dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
3. [Retina MNIST](https://medmnist.com/)

## Installation

To install the projects dependencies and create a virtual environment, make sure that your system has python (>=3.9,<3.10) and [poetry](https://python-poetry.org/) installed.

Then `cd` into the projects root directory and call: `$poetry install`
