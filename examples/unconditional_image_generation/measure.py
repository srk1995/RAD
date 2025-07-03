import argparse
import inspect
import logging
import math
import os

import numpy as np
import torch
from opencv_transforms import transforms
from torch.utils.data import DataLoader, Dataset


from metrics.lpips import lpips
from metrics.fid import fid_score
from glob import glob
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, im0_path, im1_path, mask, transform=None):
        super().__init__()
        self.im0_path = glob(f"{im0_path}/*.png")
        self.im1_path = glob(f"{im1_path}/*.png")
        self.im0_path.sort()
        self.im1_path.sort()
        self.mask = mask
        self.transform = transform


    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        self.im0 = np.array(Image.open(self.im0_path[idx]).convert("RGB"))/255
        self.im1 = np.array(Image.open(self.im1_path[idx]).convert("RGB"))/255

        self.im0 = self.transform(self.im0)
        self.im1 = self.transform(self.im1)
        return self.im0, self.im1

if __name__ == "__main__":
    loss_fn = lpips.LPIPS(pretrained=True, net='alex')
    # im0_path = '/home/srk1995/PycharmProjects/diffusers/ddpm-model-imagenet-1k-256/Local-imagenet-1k-1000-1000-0-b_bar-perlin/Inpainting images_FID_1400000/DDPM/thick/inpainted/'
    # im1_path = '/home/srk1995/PycharmProjects/diffusers/ddpm-model-imagenet-1k-256/Local-imagenet-1k-1000-1000-0-b_bar-perlin/Inpainting images_FID_1400000/DDPM/thick/original/'
    # im0_path = '/home/srk1995/PycharmProjects/diffusers/ddpm-model-imagenet-1k-256/Local-i/magenet-1k-2000-1000-0-perlin_lora_rank_16/Inpainting images_FID_comp/800000/DDPM/extreme/inpainted/'
    im0_path = "/home/srk1995/PycharmProjects/diffusers/ddpm-model-mattymchen/celeba-hq-256/box_Local-celeba-hq-2000-1000-0-perlin_lora_rank_16/Inpainting images_FID_300000/DDPM/genhalf/inpainted/"
    im1_path = '/home/srk1995/PycharmProjects/diffusers/ddpm-model-mattymchen/celeba-hq-256/box_Local-celeba-hq-2000-1000-0-perlin_lora_rank_16/Inpainting images_FID_300000/DDPM/genhalf/original/'
    # im0_path = '/home/srk1995/PycharmProjects/diffusers/ddpm-model-pcuenq/lsun-bedrooms-256/Local-lsun-bedrooms-2000-1000-0-b_bar-perlin/Inpainting images_FID_950000/DDPM/thick/inpainted/'
    # im1_path = '/home/srk1995/PycharmProjects/diffusers/ddpm-model-pcuenq/lsun-bedrooms-256/Local-lsun-bedrooms-2000-1000-0-b_bar-perlin/Inpainting images_FID_950000/DDPM/thick/original/'
    # im0_path = '/home/srk1995/PycharmProjects/MCG_diffusion-main/results_bedroom2_repaint/box/lsun_bedroom/inpainted/'
    # im1_path = '/home/srk1995/PycharmProjects/MCG_diffusion-main/results_bedroom2_repaint/box/lsun_bedroom/original/'
    # im0_path = '/home/srk1995/PycharmProjects/ddrm-master/exp/image_samples/bedroom/box/images/'
    # im1_path = '/home/srk1995/PycharmProjects/ddrm-master/exp/image_samples/bedroom/box/original/'
    mask_list = glob(f"{im0_path}/../mask/*")
    mask_list = [x.split('/')[-2] for x in mask_list]
    mask_list = set(mask_list)
    # for mask in mask_list:
    dataset = MyDataset(im0_path, im1_path, "extreme",
                        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    total_loss = []
    for i, x in enumerate(dataloader):
        im0, im1 = x
        im0, im1 = im0.to(torch.float), im1.to(torch.float)
        loss = loss_fn(im0, im1).mean()
        total_loss.append(loss.item())
    print(f"lpips: {np.mean(total_loss)} ")

    if "ffhq" in im0_path:
        # fid_value = fid.compute_fid(im0_path, dataset_name="FFHQ", dataset_res=256, dataset_split="trainval70k")
        fid_value = fid_score.calculate_fid_given_paths([
            f"{im0_path}",
            "/home/srk1995/pub/db/ffhq.npz",
            ]
            , batch_size=64, device="cuda:0", dims=2048, num_workers=0)
    elif "bedroom" in im0_path:
        fid_value = fid_score.calculate_fid_given_paths([
            f"{im0_path}",
            "/home/srk1995/PycharmProjects/lsun_bedrooms.npz"]
            , batch_size=64, device="cuda:0", dims=2048, num_workers=0)
    elif "celeb" in im0_path:
        fid_value = fid_score.calculate_fid_given_paths([
            f"{im0_path}",
            "/home/srk1995/pub/db/CelebA-HQ/celeba-256.npz"]
            , batch_size=64, device="cuda:0", dims=2048, num_workers=0)
    else:
        # fid_value = fid.compute_fid(im0_path, "/home/srk1995/pub/db/imagenet-256x256.npz", dataset_res=256, dataset_split="trainval70k")
        fid_value = fid_score.calculate_fid_given_paths([
            f"{im0_path}",
            "/home/srk1995/pub/db/ImageNet.npz"]
            , batch_size=64, device="cuda:0", dims=2048, num_workers=0)

    print(f"FID: {fid_value}")