import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from noise_scheduler import LocalDDPMScheduler

import skvideo.io

#path = "C:\\Users\\srk19\\Downloads\\LocalDiff figure\\Figures\\multiple\\"
path = "/home/srk1995/PycharmProjects/diffusers/utils/multiple_vid_mask/gen_mask/"

os.makedirs(path, exist_ok=True)

images = glob(f"/home/srk1995/PycharmProjects/diffusers/utils/multiple_vid_mask/image/*.png")
image_names = [x.split('/')[-1].split('.')[0] for x in images]

image_pairs = [
    ['0144', '0145', '0147', '0148'],
    ['0150', '0152', '0153', '0154'],
    ['0180', '0181', '0183', '0184'],
    ['0201', '0202', '0204', '0205'],
    ['0121', '0123', '0125', '0129'],
    ['0130', '0131', '0133', '0134'],
    ]

#Generate video from images
fourcc = cv2.VideoWriter_fourcc(*'XVID')
for image_pair in image_pairs:
    ## get list of images for each image in image_pair
    im_path1 = glob(f"{path}/*{image_pair[0]}*.png")
    im_path2 = glob(f"{path}/*{image_pair[1]}*.png")
    im_path3 = glob(f"{path}/*{image_pair[2]}*.png")
    im_path4 = glob(f"{path}/*{image_pair[3]}*.png")

    im_path1.sort(reverse=True)
    im_path2.sort(reverse=True)
    im_path3.sort(reverse=True)
    im_path4.sort(reverse=True)

    im1 = [cv2.imread(image) for image in im_path1[100:]]
    im2 = [cv2.imread(image) for image in im_path2[100:]]
    im3 = [cv2.imread(image) for image in im_path3[100:]]
    im4 = [cv2.imread(image) for image in im_path4[100:]]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    lines = [ax.plot([], [], lw=2)[0] for ax in axes]

    for ax in axes:
        ax.axis("off")

    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    def update(frame_index):
        for line, data in zip(lines, [im1, im2, im3, im4]):
            line.set_data(data[frame_index])
        return lines


    ani = animation.FuncAnimation(fig, update, frames=len(im1), init_func=init, blit=True)
    ani.save(f"{path}/../{image_pair[0]}_{image_pair[1]}_{image_pair[2]}_{image_pair[3]}.mp4", writer='ffmpeg', fps=30)
    # for i in range(len(im1)):
    #     ax[0].imshow(Image.open(im1[i]))
    #     ax[1].imshow(Image.open(im2[i]))
    #     ax[2].imshow(Image.open(im3[i]))
    #     ax[3].imshow(Image.open(im4[i]))
    #     for k in range(4):
    #         ax[k].axis("off")
    # fig.savefig(f"{path}/../{image_pair[0]}_{image_pair[1]}_{image_pair[2]}_{image_pair[3]}.png")
    # plt.close(fig)

