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

    min_v, max_v = 1, H
    div = (max_v - min_v) * np.random.rand(B) + min_v

    res = np.array((H / div, W / div))
    res_ceil = np.ceil(res).astype(np.int64)

    delta = (res[0] / shape[1], res[1] / shape[2])
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

        if sig == None:
            sig = int(np.random.randint(0, 21, 1))
            k = sig * 4 + 1

        if sig != 0:
            mask = GaussianBlur(kernel_size=k, sigma=(sig, sig))(mask)


        if rand:
            threshold = th.randint(0, H*W+1, (1,))
        else:
            threshold = np.random.triangular(0, 0.5, 1, B)

        if threshold.item() == 0:
            mask[i] = th.zeros_like(mask[i])
        else:
            threshold = th.kthvalue(mask[i].view(-1), k=threshold.item())[0]
            threshold = th.tensor(threshold).view(1, 1, 1)
            mask[i] = mask[i] <= threshold[:, None, None]

    return mask.float()[:, None, :, :]

