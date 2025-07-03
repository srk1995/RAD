import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from noise_scheduler import LocalDDPMScheduler
import cv2

path = "/home/pub/db/RAD_eval/video_samples"

images = glob(f"{path}/original/*.png")
masks = glob(f"{path}/mask/*.png")


save_path = f"{path}/gen_mask/"
os.makedirs(save_path, exist_ok=True)


images.sort()
masks.sort()

color = np.array([244, 232, 244]) / 255.0

for image, mask in zip(images, masks):
    filename = image.split('/')[-1].split('.')[0]

    image = np.array(Image.open(image).convert("RGB")) / 255.0
    image = image[:, :, ::-1].copy()
    mask = np.array(Image.open(mask).convert("L")) / 255.0
    mask = torch.from_numpy(mask[None, None, :])
    image = torch.from_numpy(image[None]).permute(0, 3, 1, 2)
    mask = F.interpolate(mask, 256).squeeze()
    image = F.interpolate(image, 256).squeeze().permute(1, 2, 0)
    mask = mask.numpy()


    mask_min, mask_max = np.min(mask), np.max(mask)
    mask = (mask - mask_min) / (mask_max - mask_min)
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    image = torch.tensor(image[None, :]).permute(0, 3, 1, 2)
    mask = torch.tensor(mask[None, None, :])

    steps = torch.arange(60, 500, 10)
    front_steps = torch.arange(0, 60, 2)
    rear_steps = torch.arange(500, 1000, 100)

    steps = torch.concat((front_steps, steps, rear_steps))
    scheduler = LocalDDPMScheduler()

    gen_samples = glob(f"{path}/gen_samples/{filename}*.png")
    for gen_sample in gen_samples:
        gen_sample_filename = gen_sample.split('/')[-1].split('.')[0]
        gen_sample = np.array(Image.open(gen_sample).convert("RGB")) / 255
        gen_sample = gen_sample[:, :, ::-1].copy()
        gen_sample = torch.from_numpy(gen_sample[None]).permute(0, 3, 1, 2)
        noisy_image = []
        vid = cv2.VideoWriter(f"{save_path}/{gen_sample_filename}.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, (256, 256))
        for step in steps:
            scheduler.set_variables(mask, step.view(1, 1))
            noisy_image = scheduler.add_noise(image, torch.randn_like(image), step)
            noisy_image = noisy_image[0].permute(1, 2, 0).numpy()

            noisy_image = np.clip(noisy_image, 0, 1)

            vid.write((noisy_image * 255).astype(np.uint8))
            if step == 0:
                for _ in range(5):
                    vid.write((noisy_image * 255).astype(np.uint8))


        for step in steps.flipud():
            scheduler.set_variables(mask, step.view(1, 1))
            noisy_image = scheduler.add_noise(gen_sample, torch.randn_like(gen_sample), step)
            noisy_image = noisy_image[0].permute(1, 2, 0).numpy()
            noisy_image = np.clip(noisy_image, 0, 1)
            vid.write((noisy_image * 255).astype(np.uint8))
            if step == 0:
                for _ in range(5):
                    vid.write((noisy_image * 255).astype(np.uint8))
        vid.release()

