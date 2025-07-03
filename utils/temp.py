import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn.functional as F
import os
import cv2
import skvideo.io

#%%
path = "/home/srk1995/PycharmProjects/diffusers/utils/multiple_vid_mask/gen_mask/"

os.makedirs(path, exist_ok=True)
images = glob(f"/home/srk1995/PycharmProjects/diffusers/utils/multiple_vid_mask/image/*.png")
image_names = [x.split('/')[-1].split('.')[0] for x in images]

#%%
idx = 0
images = glob(f"{path}/*{image_names[idx]}*.png")
images.sort(reverse=True)
#%%
np_images = [np.array(Image.open(image).convert("RGB")) for image in images]
np_images = np.stack(np_images, axis=0)

#%%
pause_frame_nums = np.arange(49, -1, -1)
pause_frames= np.stack([np.tile(np_images[pause_frame_num], (3*30, 1, 1, 1)) for pause_frame_num in pause_frame_nums], axis=0)
output_video_data = np.stack([np_images[:pause_frame_nums[0]], pause_frames], axis=0)
skvideo.io.vwrite(f'{path}/../video_tmp.mp4', np_images, outputdict={'-r': '30'})

