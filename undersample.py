import numpy as np
from numpy.lib.stride_tricks import as_strided
import itertools

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def cartesian_mask(shape, rate, sample_n=6, centred=True):
    assert rate > 1
    if len(shape) == 4:
        N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    elif len(shape) == 3:
        N, Nx, Ny = shape
    else:
        raise ValueError("Invalid dimension, {} given.".format(len(shape)))

    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * rate)
    n_lines = int(Nx / rate)
    assert n_lines >= sample_n
    pdf_x += lmda * 1. / Nx
    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask
