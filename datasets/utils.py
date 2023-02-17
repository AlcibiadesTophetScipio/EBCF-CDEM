import torch
from torchvision import transforms

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    version_float = float(''.join(torch.__version__.split('.')[:2]))
    if version_float>=18:
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    else:
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-value pairs.
        img: Tensor, (C, H, W)
    """
    coord = make_coord(img.shape[-2:])
    c=img.shape[0]
    img_value = img.view(c, -1).permute(1, 0)
    return coord, img_value

def resize_fn(img, size):
    return transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(img)