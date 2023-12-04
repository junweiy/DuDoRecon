import torch
import torch.fft
import os

def fft(x):
    x = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))
    return x

def ifft(x):
    x = torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho')
    return x


def torch_fft(tensor, dim):
    return fftshift(torch.fft.fftn(tensor, dim=dim), dim=dim)

def torch_ifft(tensor, dim):
    return torch.fft.ifftn(ifftshift(tensor, dim), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return torch.roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return torch.roll(x, shift, dim)

def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)