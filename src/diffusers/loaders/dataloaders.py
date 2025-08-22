# Copyright (C) 2024 Sora Kim
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from glob import glob
import cv2

import PIL.Image as Image
import numpy as np


class MakeManyMasksWrapper:
    def __init__(self, impl, variants_n=2):
        self.impl = impl
        self.variants_n = variants_n

    def get_masks(self, img):
        img = np.transpose(np.array(img), (2, 0, 1))
        return [self.impl(img)[0] for _ in range(self.variants_n)]


def process_images(src_images, config):
    config = load_yaml(config)
    variants_n = config.mask_generator_kwargs.pop('variants_n', 2)
    mask_generator = MakeManyMasksWrapper(MixedMaskGenerator(**config.mask_generator_kwargs),
                                          variants_n=1)

    max_tamper_area = config.get('max_tamper_area', 1)
    src_masks = mask_generator.get_masks(src_images.permute(1, 2, 0).numpy())
    return torch.tensor(src_masks[0]).unsqueeze(0)

class MaskDataset(Dataset):
    def __init__(self, mask_path, transform=None):
        super().__init__()
        if "box" in mask_path:
            self.mode = "box"
        elif "random" in mask_path:
            self.mode = "random"
        elif "extreme" in mask_path:
            self.mode = "extreme"
        else:
            self.mode = None
            self.mask_path = glob.glob(f"{mask_path}/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.mask_path) if self.mode is None else 50

    def __getitem__(self, idx):
        if self.mode is None:
            self.mask = Image.open(self.mask_path[idx])
            self.mask = np.array(self.mask)
            return 1 - self.transform(self.mask)[0].unsqueeze(0)
        elif self.mode == "box":
            # When given images are 256x256, generate random mask of size 128x128 box

            self.mask = torch.zeros((1, 256, 256))
            i, j = np.random.randint(0, 128, 2)
            self.mask[:, i:i + 128, j:j + 128] = 1
            return self.mask
        elif self.mode == "random":
            self.mask = torch.randn((1, 256, 256))
            threshold = np.random.randint(90, 96)
            threshold = np.percentile(self.mask, threshold)
            threshold = torch.tensor(threshold)
            self.mask = self.mask < threshold
            return self.mask.float()
        elif self.mode == "extreme":
            rand = np.random.randint(0, 2)
            if rand:
                self.mask = torch.ones(256, 256)
                self.mask[96:160, 96:160] = 0
            else:
                self.mask = torch.ones(256, 256)
                self.mask[:, 0:129] = 0
            self.mask = self.mask.unsqueeze(0).float()
            return self.mask

class InpaintDataset(Dataset):
    def __init__(self, image_data, mask_path, transform=None, lora=False):
        super().__init__()
        self.image_data = image_data
        self.mask_path = mask_path
        self.key = 'input'

        if "box" in mask_path:
            self.mode = "box"
        elif "random" in mask_path:
            self.mode = "random"
        elif "extreme" in mask_path:
            self.mode = "extreme"
        elif "medium" in mask_path:
            self.mode = "medium"
        elif "thick" in mask_path:
            self.mode = "thick"
        else:
            self.mode = None
            self.mask_path = glob.glob(f"{mask_path}/*.png")

        self.mask_len = len(self.mask_path) if self.mode is None else 1

        self.transform = transform

    def __len__(self):
        return len(self.image_data) * self.mask_len

    def __getitem__(self, idx):
        idx1 = idx // self.mask_len
        idx2 = idx % self.mask_len
        image = self.image_data[idx1][self.key]

        if self.mode is None:
            self.mask = Image.open(self.mask_path[idx2])
            self.mask = np.array(self.mask)
            self.mask = 1 - self.transform(self.mask)[0].unsqueeze(0)
        elif self.mode == "box":
            # When given images are 256x256, generate random mask of size 128x128 box

            self.mask = torch.zeros((1, 256, 256))
            i, j = np.random.randint(0, 128, 2)
            self.mask[:, i:i + 128, j:j + 128] = 1

        elif self.mode == "random":
            self.mask = torch.randn((1, 256, 256))
            threshold = np.random.randint(90, 96)
            threshold = np.percentile(self.mask, threshold)
            threshold = torch.tensor(threshold)
            self.mask = self.mask < threshold
            self.mask =  self.mask.float()
        elif self.mode == "extreme":
            self.mask = torch.ones((1, 256, 256))
            i, j = np.random.randint(0, 128, 2)
            self.mask[:, i:i + 128, j:j + 128] = 0
            # rand = np.random.randint(0, 2)
            # if rand:
            #     self.mask = torch.ones(256, 256)
            #     self.mask[96:160, 96:160] = 0
            # else:
            #     self.mask = torch.ones(256, 256)
            #     self.mask[:, 0:129] = 0
            # self.mask = self.mask.unsqueeze(0).float()

        elif self.mode in ["medium", "thick"]:
            self.mask = process_images(image,
                                       config=f"/home/srk1995/PycharmProjects/diffusers/configs/data_gen/random_{self.mode}_256.yaml")

        return image, self.mask


class GivenDataset(Dataset):
    def __init__(self, data_path, mask_type, transform=None, lora=False):
        super().__init__()
        self.image_data = f"{data_path}/{mask_type}/original/"
        self.mask_path = f"{data_path}/{mask_type}/mask/"

        self.transform = transform

    def __len__(self):
        return len(glob(f"{self.image_data}/*.png"))

    def __getitem__(self, idx):
        image = f"{self.image_data}/{idx:04d}.png"
        mask = f"{self.mask_path}/{idx:04d}.png"

        cv2.imread
        self.image = Image.open(image).convert("RGB")
        self.image = np.array(self.image)
        self.image = self.transform(self.image)
        self.mask = Image.open(mask).convert("RGB")
        self.mask = np.array(self.mask)
        self.mask = self.transform(self.mask)[0].unsqueeze(0)
        self.mask = self.mask - self.mask.min()
        self.mask = self.mask / self.mask.max()

        self.image = self.image - self.image.min()
        self.image = self.image / self.image.max()

        self.image = self.image * 2 -1

        return self.image, self.mask