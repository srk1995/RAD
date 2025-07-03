import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from noise_scheduler import LocalDDPMScheduler

#path = "C:\\Users\\srk19\\Downloads\\LocalDiff figure\\Figures\\multiple\\"

images = glob(f"/home/srk1995/pub/db/RAD_eval/text_overview/box/original/*.png")
masks = glob(f"/home/srk1995/pub/db/RAD_eval/text_overview/box/mask/*.png")

path = '/'.join(images[0].split('/')[:-2])
path += f"/gen_mask/"
os.makedirs(path, exist_ok=True)


images.sort()
masks.sort()

color = np.array([244, 232, 244]) / 255.0

for image, mask in zip(images, masks):
    filename = image.split('/')[-1].split('.')[0]

    image = np.array(Image.open(image).convert("RGB")) /255
    mask = np.array(Image.open(mask).convert("L")) / 255.0
    mask = torch.from_numpy(mask[None, None, :])
    image = torch.from_numpy(image[None]).permute(0, 3, 1, 2)
    mask = F.interpolate(mask, 256).squeeze()
    image = F.interpolate(image, 256).squeeze().permute(1, 2, 0)
    mask = mask.numpy()

    mask_min, mask_max = np.min(mask), np.max(mask)
    mask = (mask - mask_min) / (mask_max - mask_min)
    overlap_mask = np.repeat(mask[:, :, None], 3, axis=-1)

    new_image = image * (1 - overlap_mask)

    overlap_mask = overlap_mask * color
    overlap_mask = overlap_mask.astype(np.int32)

    alpha = 0.8
    new_image = new_image + image * (1 - alpha) + overlap_mask * alpha
    # new_image = image * alpha + mask * (1 - alpha)

    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis("off")
    plt.box(False)
    plt.imshow(new_image)
    # plt.show()
    plt.savefig(f"{path}/masked_{filename}", dpi=256)

    image = torch.tensor(image[None, :]).permute(0, 3, 1, 2)
    mask = torch.tensor(mask[None, None, :])


    steps = torch.arange(0, 2000, 10)
    scheduler = LocalDDPMScheduler()

    # plt.plot(range(len(scheduler.betas_cumprod_mask)), scheduler.betas_cumprod_mask)
    # ticks = [''] * len(scheduler.betas_cumprod_mask)
    # ticks[0] = '0'
    # ticks[-1] = 'T'
    # plt.xticks(range(len(scheduler.betas_cumprod_mask)), ticks)
    # plt.savefig("b_bar_to_T.eps")
    # plt.savefig("b_bar_to_T.png")

    ## TODO: forward process
    noisy_image = []
    for step in steps:
        scheduler.set_variables(mask, step.view(1, 1))
        noisy_image = scheduler.add_noise(image, torch.randn_like(image), step)
        noisy_image = noisy_image[0].permute(1, 2, 0).numpy()

        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.axis("off")
        plt.box(False)
        plt.imshow(noisy_image)

        plt.savefig(f"{path}/noisy_{step:04d}_{filename}", dpi=256)
        #plt.show()

        b_t_bar = scheduler.b_t_bar[0].permute(1, 2, 0).repeat(1, 1, 3).numpy()
        b_t_bar = b_t_bar * color
        b_t_bar[0, 0] = color
        fig = plt.figure(figsize=(1, 1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.axis("off")
        plt.box(False)
        plt.imshow(b_t_bar)


        plt.savefig(f"{path}/b_t_bar_{step:04d}_{filename}", dpi=256)


