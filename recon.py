import numpy as np

import random
import torch
import torch.nn as nn
from unet import ResUNet, torch_fft, torch_ifft
from torch.autograd import Variable

from signal_utils import mkdir

from skimage.metrics import structural_similarity

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape

    max_intensity = 1
    mse = np.sum((x - y) ** 2).astype(float) / x.size
    if mse == 0:
        return 0
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def get_ckpt_name(file_suffix, epoch, domain=''):
    if domain != '':
        domain = '_' + domain
    mkdir("./saved_models/unet_%s/" % (file_suffix))
    return "./saved_models/unet_%s/unet_%s%s_%d.pth" % (file_suffix, file_suffix, domain, epoch)


def get_concated_tensor(trans_model, imgs_t1, imgs_t2u, mode='recon'):
    if mode == 'trans':
        concated = imgs_t1
    elif mode == 'recon':
        if trans_model is not None:
            trans_t2 = trans_model(imgs_t1, None).clone().detach()
            concated = torch.cat((trans_t2, imgs_t2u), 1)
        else:
            concated = torch.cat((imgs_t1, imgs_t2u), 1)
    else:
        raise ValueError
    return concated


def pred_t2_img(model, concated, mode, img_mask, domain, u_mask=None, factor=None):
    if mode == 'recon':
        gen_t2 = model(concated, u_mask)
    elif mode == 'trans':
        assert domain in [0, 1]
        gen_t2 = model(concated, u_mask)
    elif mode == 'reg':
        if domain == 0:
            moving = torch_ifft(concated[:, :1] + 1j * concated[:, 1:2], (-2, -1))
            fixed = torch_ifft(concated[:, 2:3] + 1j * concated[:, 3:], (-2, -1))
            moving = torch.concat((moving.real, moving.imag), 1)
            moving = torch.concat((fixed.real, fixed.imag), 1)
            gen_t2, _ = model(fixed, moving)
        elif domain == 1:
            moving = concated[:, :2]
            fixed = concated[:, 2:]
            gen_t2, _ = model(fixed, moving)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # post processing to remove generated image background of query modality
    if img_mask is not None:
        if domain in [1, 2] or mode == 'reg':
            gen_t2 *= img_mask
            if domain == 0 and mode == 'reg':
                gen_t2 = torch_fft(gen_t2[:, :1] + 1j * gen_t2[:, 1:], (-2, -1))
                gen_t2 = torch.concat((gen_t2.real, gen_t2.imag), 1)
        elif domain == 0: # k-space
            gen_t2 = gen_t2 * factor
            gen_t2_c = torch.complex(gen_t2[:, :1, ...], gen_t2[:, 1:, ...])
            gen_t2_img = torch_ifft(gen_t2_c, (-2, -1))
            gen_t2_img *= img_mask
            gen_t2 = torch_fft(gen_t2_img, (-2, -1))
            gen_t2 = torch.concat((gen_t2.real, gen_t2.imag), 1)
            gen_t2 = gen_t2 / factor
    return gen_t2


def train_recon(model, trans_model, dataloader, val_dataloader, c_epoch, n_epochs, file_suffix, u_mask, domain, mode='recon', alpha=0.1):
    # Losses
    criterion_unet = nn.L1Loss()
    if domain == 2:
        model_i = model[0]
        model_k = model[1]
        if torch.cuda.is_available():
            model_i = model_i.cuda()
            model_k = model_k.cuda()
        optimizer_unet_i = torch.optim.Adam(model_i.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_unet_k = torch.optim.Adam(model_k.parameters(), lr=0.0002, betas=(0.5, 0.999))
        if c_epoch != 0:
            model_i.load_state_dict(torch.load(get_ckpt_name(file_suffix, c_epoch - 1, domain='i')))
            model_k.load_state_dict(torch.load(get_ckpt_name(file_suffix, c_epoch - 1, domain='k')))
        model_i.train()
        model_k.train()
    else:
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer_unet = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

        if c_epoch != 0:
            model.load_state_dict(torch.load(get_ckpt_name(file_suffix, c_epoch - 1)))
        model.train()
    if torch.cuda.is_available():
        criterion_unet = criterion_unet.cuda()

    step = c_epoch * len(dataloader)

    for epoch in range(c_epoch, c_epoch + n_epochs):
        val_step = step

        for i, imgs in enumerate(dataloader):

            imgs_t1, imgs_t2, imgs_t2u = get_data_pair(imgs, u_mask)
            concated = get_concated_tensor(trans_model, imgs_t1, imgs_t2u, mode)

            ######################
            # Beginning of U-Net #
            ######################
            if domain == 2:
                model_i.zero_grad()
                model_k.zero_grad()
                gen_t2_i = pred_t2_img(model_i, None, None, imgs_t1[:, :1, ...], 'trans')
                gen_t2_k = pred_t2_img(model_k, None, None, imgs_t1[:, 1:, ...], 'trans')
                loss_i = criterion_unet(imgs_t2[:, :1, ...], gen_t2_i)
                loss_k = criterion_unet(imgs_t2[:, 1:, ...], gen_t2_k)
                gen_t2_ik = torch_fft(gen_t2_i, (-2, -1))
                gen_t2_ik = torch.concat((gen_t2_ik.real, gen_t2_ik.imag), dim=1)
                loss_ik = criterion_unet(imgs_t2[:, 1:, ...], gen_t2_ik)
                gen_t2_ki = abs(torch_ifft(torch.complex(gen_t2_k[:, :1], gen_t2_k[:, 1:]), (-2, -1)))
                loss_ki = criterion_unet(imgs_t2[:, :1, ...], gen_t2_ki)
                loss_unet = loss_i + loss_k + alpha * (loss_ik + loss_ki)
                loss_unet.backward()
                optimizer_unet_i.step()
                optimizer_unet_k.step()
            else:
                model.zero_grad()
                gen_t2 = pred_t2_img(model, concated, u_mask, imgs_t1, mode)
                loss_unet = criterion_unet(imgs_t2, gen_t2)
                loss_unet.backward()
                optimizer_unet.step()

            t2re_psnr = psnr(imgs_t2.cpu().numpy()[0,0], gen_t2.detach().cpu().numpy()[0,0])
            if u_mask is not None:
                t2u_psnr = psnr(imgs_t2.cpu().numpy()[0, 0], imgs_t2u.cpu().numpy()[0, 0])
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [U-Net loss: %f] [T2u PSNR: %.4f] [T2re PSNR: %.4f] \n"
                    % (epoch, n_epochs, i, len(dataloader), loss_unet.item(), t2u_psnr, t2re_psnr)
                )
            else:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [U-Net loss: %f] [T2re PSNR: %.4f] \n"
                    % (epoch, n_epochs, i, len(dataloader), loss_unet.item(), t2re_psnr)
                )

            if mode == 'trans':
                print(
                        "[Epoch %d/%d] [Batch %d/%d] [U-Net loss: %f]"
                        % (epoch, n_epochs, i, len(dataloader), loss_unet.item())
                    )

            step += 1

        if epoch % 10 == 0:
            # Save model checkpoints
            if domain == 2:
                torch.save(model_i.state_dict(), get_ckpt_name(file_suffix, epoch, domain='i'))
                torch.save(model_k.state_dict(), get_ckpt_name(file_suffix, epoch, domain='k'))
            else:
                torch.save(model.state_dict(), get_ckpt_name(file_suffix, epoch))

    if domain == 2:
        return [model_i, model_k]
    else:
        return model


def c_k2r_i(k_data):
    if k_data is None:
        return None
    if not torch.is_complex(k_data):
        k_data_c = k_data[:, 0, ...] + 1j * k_data[:, 1, ...]
    else:
        k_data_c = k_data
    if isinstance(k_data, torch.Tensor):
        iffted_k = torch_ifft(k_data_c, (-2, -1)).detach().cpu().numpy()
        return np.concatenate((iffted_k.real, iffted_k.imag), axis=1)
    elif isinstance(k_data, np.ndarray):
        return NotImplementedError
    else:
        raise NotImplementedError

def get_psnr_ssim(imgs_t2, pred_t2, imgs_t2u):
    if isinstance(imgs_t2, torch.Tensor):
        imgs_t2_np = imgs_t2.cpu().detach().numpy()
    else:
        imgs_t2_np = imgs_t2
    if isinstance(pred_t2, torch.Tensor):
        pred_t2_np = pred_t2.cpu().detach().numpy()
    else:
        pred_t2_np = pred_t2
    pred_t2_np[np.where(imgs_t2_np == 0)] = 0
    t2u_psnr = 0
    if imgs_t2u is not None:
        if isinstance(imgs_t2u, torch.Tensor):
            imgs_t2u_np = imgs_t2u.cpu().detach().numpy() * (imgs_t2_np != 0)
        else:
            imgs_t2u_np = imgs_t2u * (imgs_t2_np != 0)
        t2u_psnr = psnr(imgs_t2u_np, imgs_t2_np)
        if t2u_psnr == 0:
            return 0, 0, 0
    pred_psnr = psnr(pred_t2_np, imgs_t2_np)
    pred_ssim = 0

    for sl in range(pred_t2_np.shape[0]):
        if pred_t2_np.shape[1] == 1:
            pred_ssim += structural_similarity(pred_t2_np[sl], imgs_t2_np[sl])
        else:
            pred_t2_t = np.transpose(pred_t2_np[sl], [1, 2, 0])
            imgs_t2_t = np.transpose(imgs_t2_np[sl], [1, 2, 0])
            pred_ssim += structural_similarity(pred_t2_t, imgs_t2_t, multichannel=True)
    pred_ssim /= pred_t2_np.shape[0]
    return t2u_psnr, pred_psnr, pred_ssim

def get_data_pair(imgs, u_mask=None, domain=None):
    if imgs[0].shape[1] == 1 and torch.is_complex(imgs[0]): # complex data
        imgs_t1 = torch.concat((imgs[0].real, imgs[0].imag), 1)
        imgs_t1 = Variable(imgs_t1.type(Tensor))
        imgs_t2 = torch.concat((imgs[1].real, imgs[1].imag), 1)
        imgs_t2 = Variable(imgs_t2.type(Tensor))
    elif imgs[0].shape[1] == 2 and torch.is_complex(imgs[0]):
        imgs_t1_i = torch.concat((imgs[0][:, :1].real, imgs[0][:, :1].imag), 1)
        imgs_t1_k = torch.concat((imgs[0][:, 1:].real, imgs[0][:, 1:].imag), 1)
        imgs_t1 = torch.concat((imgs_t1_i, imgs_t1_k), 1)
        imgs_t1 = Variable(imgs_t1.type(Tensor))
        imgs_t2_i = torch.concat((imgs[1][:, :1].real, imgs[1][:, :1].imag), 1)
        imgs_t2_k = torch.concat((imgs[1][:, 1:].real, imgs[1][:, 1:].imag), 1)
        imgs_t2 = torch.concat((imgs_t2_i, imgs_t2_k), 1)
        imgs_t2 = Variable(imgs_t2.type(Tensor))
    else:
        imgs_t1 = Variable(imgs[0].type(Tensor))
        imgs_t2 = Variable(imgs[1].type(Tensor))
    imgs_t2u = None
    try:
        t1_mask = Variable(imgs[3][0].type(Tensor))
        t2_mask = Variable(imgs[3][1].type(Tensor))
    except IndexError:
        t1_mask = None
        t2_mask = None

    if u_mask is not None:
        assert domain is not None
        if domain == 0:
            ku = (imgs_t2[:, :1] + 1j * imgs_t2[:, 1:]) * u_mask
        elif domain == 1:
            imgs_t2_c = imgs_t2[:, :1] + 1j * imgs_t2[:, 1:]
            ku = torch_fft(imgs_t2_c, (-2, -1)) * u_mask
        elif domain == 2:
            ku = (imgs_t2[:, 2:3] + 1j * imgs_t2[:,3:]) * u_mask
        t2u = torch_ifft(ku, (-2, -1))
        if domain == 0:
            imgs_t2u = torch.concat((ku.real, ku.imag), axis=1)
        elif domain == 1:
            imgs_t2u = torch.concat((t2u.real, t2u.imag), axis=1)
        elif domain == 2:
            imgs_t2u = torch.concat((t2u.real, t2u.imag, ku.real, ku.imag), axis=1)
        if torch.cuda.is_available():
            imgs_t2u = imgs_t2u.cuda()
        imgs_t2u = Variable(imgs_t2u.type(Tensor))


    return imgs_t1, imgs_t2, imgs_t2u, t1_mask, t2_mask
