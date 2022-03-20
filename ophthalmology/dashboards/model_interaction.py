# -*- coding: utf-8 -*-
""" Seminar in AI: Contrastive Learning for Ophthalmology
Laurenz Hundgeburth
"""

import einops
import hydra
import matplotlib.pyplot as plt
import streamlit as st
import torch
from torchvision import transforms

from ophthalmology.data import sets

st.markdown("# Fovea and Optical Disk Localization")

with hydra.initialize(config_path="../conf"):
    config = hydra.compose(
        config_name="train_config.yaml",
        overrides=["experiment=sup_idrd_localization", "environment=laptop"],
    )

    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(
        torch.load("registry/pytorch/sup_idrd_localization.pt")
    )
    model.eval()

    test_transforms: torch.nn.Module = hydra.utils.instantiate(
        config.test_transforms
    )

    datamodule = hydra.utils.instantiate(
        config.datamodule,
        train_transform=transforms.ToTensor(),
        test_transform=transforms.ToTensor(),
    )

    dataset = datamodule.data_set  # datamodule.test_dataset

    image_size = st.slider("image size of model input", 28, 1000, 256)

    prep = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(
                [0.3211, 0.2243, 0.1602], [0.2617, 0.1825, 0.1308]
            ),
        ]
    )

idx = st.number_input("sample index", 0, len(dataset))


@st.cache()
def get_item(index):
    return dataset[index]


img, labels = get_item(idx)

plot = lambda img: einops.rearrange(img, "c w h -> w h c")


@st.cache()
def pred(img):
    return model(prep(img))


preds = pred(img)
preds = [preds[0, i] for i in range(4)]


@st.cache()
def norm2pixel(norm):
    return (
        int(norm[0].item() * 4288),
        int(norm[1].item() * 2848),
        int(norm[2].item() * 4288),
        int(norm[3].item() * 2848),
    )


fx, fy, dx, dy = norm2pixel(labels)
fpx, fpy, dpx, dpy = norm2pixel(preds)

fig = plt.figure()
plt.plot(fx, fy, marker="x", color="green")
plt.plot(fpx, fpy, marker="x", color="red")
plt.plot(dx, dy, marker="x", color="blue")
plt.plot(dpx, dpy, marker="x", color="orange")
plt.legend(["fovea true", "fovea pred", "disk true", "disk pred"])
plt.plot([fx, fpx], [fy, fpy], color="white", linewidth=1)
plt.plot([dx, dpx], [dy, dpy], color="white", linewidth=1)
plt.imshow(plot(img))

st.write(fig)
