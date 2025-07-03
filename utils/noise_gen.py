import torch as th
import numpy as np
from torchvision.transforms import GaussianBlur
import time



def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)

def generate_perlin_noise_2d(
        shape, rand=True, sig=None, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    B, H, W = shape
    mask = th.zeros(B, H, W)

    # # log scale uniform
    # div = np.random.rand(np.log2(H) - 4, np.log2(H), B)
    # min_v, max_v = 0, np.log2(H)
    # div = (max_v - min_v) * np.random.rand(B) + min_v
    # div = 2 ** div

    # linear uniform continuous
    # div = np.random.rand(H//16, H, B)
    min_v, max_v = 1, H
    div = (max_v - min_v) * np.random.rand(B) + min_v

    res = np.array((H / div, W / div))
    res_ceil = np.ceil(res).astype(np.int64)

    delta = (res[0] / shape[1], res[1] / shape[2])
    for i in range(B):
        x, y = th.arange(0, res[0][i], delta[0][i]), th.arange(0, res[1][i], delta[1][i])
        # x, y = th.arange(0, shape[1]) * delta[0][i], th.arange(0, shape[2]) * delta[1][i]
        grid = th.stack(th.meshgrid(x, y), dim=-1) % 1
        # Gradients
        angles = 2*th.pi*th.rand(1, res_ceil[0][i] + 1, res_ceil[1][i] + 1)
        gradients = th.stack((th.cos(angles), th.sin(angles)), dim=-1)
        if tileable[0]:
            gradients[:,:,-1,:] = gradients[:,:,0,:]
        if tileable[1]:
            gradients[:,:,:,-1] = gradients[:,:,:,0]

        x0 = x.floor().to(th.int)
        x1 = x0 + 1
        y0 = y.floor().to(th.int)
        y1 = y0 + 1

        # (a)
        # start = time.time()
        # g00 = gradients[:, x0][:, :, y0]
        # g10 = gradients[:, x1][:, :, y0]
        # g01 = gradients[:, x0][:, :, y1]
        # g11 = gradients[:, x1][:, :, y1]
        # print(f"a: {time.time()-start}")

        #(b)
        # start = time.time()
        tmp = gradients[:, x0]
        g00 = tmp[:, :, y0]
        g01 = tmp[:, :, y1]
        tmp = gradients[:, x1]
        g10 = tmp[:, :, y0]
        g11 = tmp[:, :, y1]
        # print(f"b: {time.time() - start}")


        # gradients = gradients.repeat_interleave(div[0][i], 1).repeat_interleave(div[1][i], 2)
        # g00 = gradients[:, :-div[0][i], :-div[1][i], :]
        # g10 = gradients[:, div[0][i]:, :-div[1][i], :]
        # g01 = gradients[:, :-div[0][i], div[1][i]:, :]
        # g11 = gradients[:, div[0][i]:, div[1][i]:, :]
        # # Ramps

        n00 = th.sum(th.stack((grid[None,:,:,0]  , grid[None,:,:,1]  ), dim=-1) * g00, -1)
        n10 = th.sum(th.stack((grid[None,:,:,0]-1, grid[None,:,:,1]  ), dim=-1) * g10, -1)
        n01 = th.sum(th.stack((grid[None,:,:,0]  , grid[None,:,:,1]-1), dim=-1) * g01, -1)
        n11 = th.sum(th.stack((grid[None,:,:,0]-1, grid[None,:,:,1]-1), dim=-1) * g11, -1)

        # Interpolation
        t = interpolant(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        mask[i] = th.sqrt(th.tensor([2]))*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

        # mask -= mask.view(B, -1).min(dim=-1)[0].view(B, 1, 1)
        # mask /= mask.view(B, -1).max(dim=-1)[0].view(B, 1, 1)

        if sig == None:
            sig = int(np.random.randint(0, 21, 1))
            k = sig * 4 + 1

        if sig != 0:
            mask = GaussianBlur(kernel_size=k, sigma=(sig, sig))(mask)
        # mask = th.clip(mask, 0, 1)

        if rand:
            threshold = th.randint(0, H*W+1, (1,))
        else:
            threshold = np.random.triangular(0, 0.5, 1, B)
        # threshold = np.percentile(mask[i].view(-1), threshold*100)
        if threshold.item() == 0:
            mask[i] = th.zeros_like(mask[i])
        else:
            threshold = th.kthvalue(mask[i].view(-1), k=threshold.item())[0]
            threshold = th.tensor(threshold).view(1, 1, 1)
            mask[i] = mask[i] <= threshold[:, None, None]

    return mask.float()[:, None, :, :]

def generate_perlin_noise_2d_sample(
        shape, threshold, scale, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    B, H, W = shape
    mask = th.zeros(B, H, W)

    res = np.array((H / scale, W / scale))
    res_ceil = np.ceil(res).astype(np.int64)

    delta = (res[0] / H, res[1] / W)
    for i in range(B):
        x, y = th.arange(0, res[0][i], delta[0][i]), th.arange(0, res[1][i], delta[1][i])
        grid = th.stack(th.meshgrid(x, y), dim=-1) % 1
        # Gradients
        angles = 2*th.pi*th.rand(1, res_ceil[0][i] + 1, res_ceil[1][i] + 1)
        gradients = th.stack((th.cos(angles), th.sin(angles)), dim=-1)
        if tileable[0]:
            gradients[:,:,-1,:] = gradients[:,:,0,:]
        if tileable[1]:
            gradients[:,:,:,-1] = gradients[:,:,:,0]

        x0 = x.floor().to(th.int)
        x1 = x0 + 1
        y0 = y.floor().to(th.int)
        y1 = y0 + 1


        tmp = gradients[:, x0]
        g00 = tmp[:, :, y0]
        g01 = tmp[:, :, y1]
        tmp = gradients[:, x1]
        g10 = tmp[:, :, y0]
        g11 = tmp[:, :, y1]

        n00 = th.sum(th.stack((grid[None,:,:,0]  , grid[None,:,:,1]  ), dim=-1) * g00, -1)
        n10 = th.sum(th.stack((grid[None,:,:,0]-1, grid[None,:,:,1]  ), dim=-1) * g10, -1)
        n01 = th.sum(th.stack((grid[None,:,:,0]  , grid[None,:,:,1]-1), dim=-1) * g01, -1)
        n11 = th.sum(th.stack((grid[None,:,:,0]-1, grid[None,:,:,1]-1), dim=-1) * g11, -1)

        # Interpolation
        t = interpolant(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        mask[i] = th.sqrt(th.tensor([2]))*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

    mask -= mask.view(B, -1).min(dim=-1)[0].view(B, 1, 1)
    mask /= mask.view(B, -1).max(dim=-1)[0].view(B, 1, 1)
    mask = th.clip(mask, 0, 1)

    threshold = np.percentile(mask, threshold*100)
    threshold = th.tensor(threshold)
    mask = mask > threshold[:, None, None]
    return mask.float()[:, None, :, :]

def gen_mask_parallel(shape, expectation=0.2, p=1.3, k=9, sig=2, device=None, center_point=None):
    B, C, H, W = shape
    N = 100 # number of blobs
    mask = th.zeros(B, H, W)

    y, x = th.linspace(0, 1, H).to(device), th.linspace(0, 1, W).to(device)
    y, x = th.meshgrid(y, x)  # H x W

    if center_point is None:
        center_point = th.rand((B, N, 2)).to(device)  # B x N x 2
    else:
        center_point = center_point.to(device)
    radius = th.distributions.exponential.Exponential(rate=1 / 0.2).sample([B, N]).to(device)  # B x N
    d = th.minimum(th.pi * radius ** 2, th.ones_like(radius)).to(device)  # B x N

    # if p <= 1:
    #     l = 1 - (1 - d) * p
    # else:
    #     l = d / p

    l = d ** p

    prob = th.rand((B, N), device=device) <= l
    idx = th.arange(prob.shape[1], 0, -1, device=prob.device)
    prob_bool = prob * idx
    indices = th.argmax(prob_bool, 1, keepdim=True)

    # index to mask
    m = th.ones(B, N, device=prob.device)
    m[th.arange(B), indices.squeeze()] = 0.
    m = m.cumprod(dim=1)

    # apply indexing mask
    radius = radius * m
    # distance: B x H x W x N
    distance = th.sqrt((x[None, :, :, None] - center_point[:, None, None, :, 0]) ** 2 + (
                y[None, :, :, None] - center_point[:, None, None, :, 1]) ** 2)
    # distance = distance * m[:, None, None, :]

    mask = distance < radius[:, None, None, :]
    mask = th.any(mask, dim=-1).float()

    if sig == None:
        sig_p = np.random.randint(0, 2)
        if sig_p:
            sig = int(th.distributions.exponential.Exponential(rate=expectation).sample([1]).item())
            k = sig * 4 + 1
        else:
            sig = 0
            k = 1

    if sig != 0:
        mask = GaussianBlur(kernel_size=k, sigma=(sig, sig))(mask)
    mask = th.clip(mask, 0, 1)
    return mask.unsqueeze(1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    B, C, H, W = 500, 1, 256, 256
    perlin_list = []
    fig_names = []
    ratios = []
    for _ in range(10):
        noise = generate_perlin_noise_2d((B, H, W))
        ratios.append(noise.sum(dim=(1, 2, 3)) / (H * W))
    ratios = th.concat(ratios)
    # scales = np.array([2, 8, 16, 64, 128])
    # ratios = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    # for scale in scales:
    #     for ratio in ratios:
    #         ratio = np.repeat(ratio, B, axis=0)
    #         scale = np.repeat(scale, B, axis=0)
    #         noise = generate_perlin_noise_2d_sample((B, H, W), ratio, scale)
    #         perlin_list.append(noise)
    #         fig_names.append(f"scale_{scale[0]}_ratio_{ratio[0]}")
    # perlin_list = th.cat(perlin_list)


    fig, ax = plt.subplots(len(scales), len(ratios))
    for i in range(len(perlin_list)):
        ax[i//len(ratios), i%len(ratios)].imshow(perlin_list[i][0, 0].numpy())
        ax[i//len(ratios), i%len(ratios)].axis('off')
    plt.show()
    plt.close()

    import os
    os.makedirs("perlin_noise", exist_ok=True)
    for i in range(len(perlin_list)):
        # fig, ax = plt.subplots(1, 1, gridspec_kw={'wspace':0, 'hspace': 0}, squeeze=True)
        # plt.axis("off")
        # ax.imshow(perlin_list[i,0].numpy())
        m = perlin_list[i][0, 0].numpy().reshape(H, W, 1) * np.array([244, 232, 244]) / 255.0
        plt.imsave(f"perlin_noise/{fig_names[i]}.png", m)