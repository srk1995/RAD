import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import os

import matplotlib.pyplot as plt

path = "/home/srk1995/pub/db/RAD_eval/Imagenet/DDPM/"
# path = "C:\\Users\\srk19\\Downloads\\LocalDiff figure\\Figures\\multiple\\"

save_path = '/'.join(path.split('/')[:-2])
save_path += f"/qualitative_results/masked_sample"

os.makedirs(save_path, exist_ok=True)


images = glob(f"{path}/*/original/*.png")
masks = glob(f"{path}/*/mask/*.png")
images.sort()
masks.sort()

color = np.array([244, 232, 244]) / 255.0
for i in range(len(images)):
    base_name = images[i].split('/')[-1].split('.')[0]
    mask_type = masks[i].split('/')[-3]
    os.makedirs(f"{save_path}/{mask_type}", exist_ok=True)
    image = np.array(Image.open(images[i]).convert("RGB")) / 255.
    mask = np.array(Image.open(masks[i]).convert("L")) / 255.0
    mask = torch.from_numpy(mask[None, None, :])
    image = torch.from_numpy(image[None]).permute(0, 3, 1, 2)
    mask = F.interpolate(mask, 256).squeeze()
    image = F.interpolate(image, 256).squeeze().permute(1, 2, 0)
    mask = mask.numpy()

    mask_min, mask_max = np.min(mask), np.max(mask)
    mask = (mask - mask_min) / (mask_max - mask_min)
    mask = np.repeat(mask[:, :, None], 3, axis=-1)

    new_image = image * (1 - mask)

    mask = mask * color

    alpha = 0.8
    new_image = new_image + image * (1 - alpha) * mask + mask * alpha
    # new_image = image * alpha + mask * (1 - alpha)

    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis("off")
    plt.box(False)
    plt.imshow(new_image)
    #plt.show()
    plt.savefig(f"{save_path}/{mask_type}/masked_{base_name}", dpi=500)


