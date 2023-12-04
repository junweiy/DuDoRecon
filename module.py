import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from recon import get_ckpt_name, get_data_pair, pred_t2_img, torch_fft, torch_ifft, c_k2r_i, get_psnr_ssim

from undersample import cartesian_mask

from unet import ResUNet, RegNet

import numpy as np

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

class ModelModule:

    def __init__(self, domain, file_suffix, task, un_rate, alpha=0.1, lr=0.0002, k_weight=1000., curr_epoch=0,
                 recon_modal='t1t2u', query_modal='', trans_module=None,
                 reg_module=None, trained_dataloader=None):
        self.domain = domain
        self.task = task
        assert self.task in ['trans', 'reg', 'recon']
        self.recon_modal = recon_modal
        self.model = self.init_model()
        modal_file_prefix = '_' + recon_modal if recon_modal != '' else ''
        self.file_suffix = file_suffix + '_alpha_' + str(alpha) + (
            '_kweight_' + str(k_weight) if self.domain == 2 else '') + modal_file_prefix
        self.un_rate = un_rate
        self.query_modal = query_modal

        # Load other modules if for registration and reconstruction
        self.trans_module = trans_module
        self.reg_module = reg_module

        # training
        self.lr = lr
        self.curr_epoch = curr_epoch
        self.k_weight = k_weight
        if self.curr_epoch > 0:
            self.load_model_weight(self.curr_epoch)
        self.alpha = alpha
        self.optimizer, self.criterion = self.get_model_optims_loss()

    def init_model(self):
        # we only use different losses to train registration network but input is always image
        if self.task == 'reg':
            model = [RegNet(4, self.domain)]
        if self.domain in [0, 1]:  # k-space
            if self.task == 'trans':
                model = [ResUNet(2, 2, 'i' if self.domain == 1 else 'k')]
            elif self.task == 'reg':
                model = [RegNet(4, self.domain)]
            elif self.task == 'recon':
                input_channel_num = 2 if self.recon_modal == 't2u' else 4
                model = [ResUNet(input_channel_num, 2, 'i' if self.domain == 1 else 'k')]
        elif self.domain == 2:
            if self.task == 'trans':
                model_i = ResUNet(2, 2, 'i')
                model_k = ResUNet(2, 2, 'k')
                model = [model_i, model_k]
            elif self.task == 'reg':
                model_i = RegNet(4, 1)
                model_k = RegNet(4, 0)
                model = [model_i, model_k]
            elif self.task == 'recon':
                input_channel_num = 2 if self.recon_modal == 't2u' else 4
                model_i = ResUNet(input_channel_num, 2, 'i')
                model_k = ResUNet(input_channel_num, 2, 'k')
                model = [model_i, model_k]

        # check if gpu is available
        model = [i.cuda() if torch.cuda.is_available() else i for i in model]

        return model

    def load_model_weight(self, epoch):
        self.curr_epoch = epoch + 1
        # temporary device switch
        if not torch.cuda.is_available():
            if is_dudo(self.domain):
                self.model[0].load_state_dict(
                    torch.load(get_ckpt_name(self.file_suffix, epoch, domain='dual_i', task=self.task),
                               map_location=torch.device('cpu')))
                self.model[1].load_state_dict(
                    torch.load(get_ckpt_name(self.file_suffix, epoch, domain='dual_k', task=self.task),
                               map_location=torch.device('cpu')))
            else:
                self.model[0].load_state_dict(
                    torch.load(
                        get_ckpt_name(self.file_suffix, epoch, domain='i' if self.domain == 1 else 'k', task=self.task),
                        map_location=torch.device('cpu')))
        else:
            if is_dudo(self.domain):
                self.model[0].load_state_dict(
                    torch.load(get_ckpt_name(self.file_suffix, epoch, domain='dual_i', task=self.task)))
                self.model[1].load_state_dict(
                    torch.load(get_ckpt_name(self.file_suffix, epoch, domain='dual_k', task=self.task)))
            else:
                self.model[0].load_state_dict(
                    torch.load(
                        get_ckpt_name(self.file_suffix, epoch, domain='i' if self.domain == 1 else 'k',
                                      task=self.task)))

    def save_model_weight(self, epoch):
        if is_dudo(self.domain):
            torch.save(self.model[0].state_dict(),
                       get_ckpt_name(self.file_suffix, epoch, domain='dual_i', task=self.task))
            torch.save(self.model[1].state_dict(),
                       get_ckpt_name(self.file_suffix, epoch, domain='dual_k', task=self.task))
        else:
            torch.save(self.model[0].state_dict(),
                       get_ckpt_name(self.file_suffix, epoch, domain='i' if self.domain == 1 else 'k', task=self.task))

    def get_model_optims_loss(self):
        criterion = nn.MSELoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()

        if is_dudo(self.domain):
            optimizer_i = torch.optim.Adam(self.model[0].parameters(), lr=self.lr, betas=(0.5, 0.999),
                                           weight_decay=1e-5, eps=1e-4)
            optimizer_k = torch.optim.Adam(self.model[1].parameters(), lr=self.lr, betas=(0.5, 0.999),
                                           weight_decay=1e-5, eps=1e-4)

            return [optimizer_i, optimizer_k], criterion
        else:
            optimizer = torch.optim.Adam(self.model[0].parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=1e-5,
                                         eps=1e-4)
            return optimizer, criterion

    def train(self, dataloader, num_epochs, val_dataloader=None):
        train_writer = SummaryWriter(comment=self.file_suffix + '_train')
        val_writer = SummaryWriter(comment=self.file_suffix + '_val')

        # use cartesian undersampling mask
        u_mask = cartesian_mask((1, dataloader.dataset.dim[0], dataloader.dataset.dim[1]), self.un_rate)[0,]
        u_mask = Tensor(u_mask)
        # initialize model
        if is_dudo(self.domain):
            self.model[0].train()
            self.model[1].train()
        else:
            self.model[0].train()

        step = self.curr_epoch * len(dataloader)

        for epoch in range(self.curr_epoch, self.curr_epoch + num_epochs):
            val_step = step

            for i, imgs in enumerate(dataloader):

                imgs_t1, imgs_t2, imgs_t2u, t1_mask, t2_mask = get_data_pair(imgs, u_mask, self.domain)

                model_input, model_gt = self.get_training_pairs(imgs_t1, imgs_t2, imgs_t2u, u_mask, (t1_mask, t2_mask))

                if is_dudo(self.domain):
                    self.model[0].zero_grad()
                    self.model[1].zero_grad()

                    loss_list = self.get_loss_value(model_input, model_gt, u_mask,
                                                    t1_mask if self.task == 'trans' else t2_mask)
                    loss_list[0].backward()

                else:
                    self.model[0].zero_grad()

                    loss_list = self.get_loss_value(model_input, model_gt, u_mask,
                                                    t1_mask if self.task == 'trans' else t2_mask)
                    loss_list[0].backward()

                self.update_loss()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                    % (epoch, self.curr_epoch + num_epochs, i, len(dataloader), loss_list[0].item())
                )

                self.write_summary(train_writer, step, loss_list)

                step += 1

            if val_dataloader is not None:
                loss_total = 0
                for i, imgs in enumerate(val_dataloader):
                    imgs_t1, imgs_t2, imgs_t2u, t1_mask, t2_mask = get_data_pair(imgs, u_mask, self.domain)

                    model_input, model_gt = self.get_training_pairs(imgs_t1, imgs_t2, imgs_t2u, u_mask,
                                                                    (t1_mask, t2_mask))

                    loss_list = self.get_loss_value(model_input, model_gt, u_mask,
                                                    t1_mask if self.task == 'trans' else t2_mask)
                    loss_total += loss_list[0]
                self.write_summary(val_writer, epoch, loss_list)

            if epoch >= self.curr_epoch + num_epochs - 10:
                # Save model checkpoints
                self.save_model_weight(epoch)

    def update_loss(self):
        if self.task == 'trans':
            if self.domain == 2:
                self.optimizer[0].step()
                self.optimizer[1].step()
            else:
                self.optimizer.step()
        if self.task == 'reg':
            if self.domain == 2:
                self.optimizer[0].step()
                self.optimizer[1].step()
            else:
                self.optimizer.step()
        if self.task == 'recon':
            if self.domain == 2:
                self.optimizer[0].step()
                self.optimizer[1].step()
            else:
                self.optimizer.step()

    def pred(self, img_input, u_mask, img_mask):
        if self.domain == 2:
            if self.task == 'trans' or (self.task == 'recon' and self.recon_modal == 't2u'):
                im_input = img_input[:, :2]
                k_input = img_input[:, 2:]
            else:
                im_input = torch.concat((img_input[:, :2], img_input[:, 4:6]), 1)
                k_input = torch.concat((img_input[:, 2:4], img_input[:, 6:]), 1)
            gen_t2_i = pred_t2_img(self.model[0], im_input, self.task, img_mask, 1, u_mask)
            gen_t2_k = pred_t2_img(self.model[1], k_input, self.task, img_mask, 0, u_mask)
            gen_t2_ik = torch_fft(gen_t2_i[:, :1] + 1j * gen_t2_i[:, 1:], (-2, -1))
            gen_t2_ik = torch.concat((gen_t2_ik.real, gen_t2_ik.imag), dim=1)
            gen_t2_ki = torch_ifft(gen_t2_k[:, :1] + 1j * gen_t2_k[:, 1:], (-2, -1))
            gen_t2_ki = torch.concat((gen_t2_ki.real, gen_t2_ki.imag), dim=1)
            return [gen_t2_i, gen_t2_k, gen_t2_ik, gen_t2_ki]
        else:
            gen_t2 = pred_t2_img(self.model[0], img_input, self.task, img_mask, self.domain, u_mask)
            return gen_t2

    def get_loss_value(self, img_input, img_gt, u_mask, img_mask):
        if self.domain == 2:
            gen_t2_i, gen_t2_k, gen_t2_ik, gen_t2_ki = self.pred(img_input, u_mask, img_mask)

            loss_i = self.criterion(img_gt[:, :2, ...], gen_t2_i)
            loss_k = self.criterion(img_gt[:, 2:, ...], gen_t2_k)
            loss_ik = self.criterion(img_gt[:, 2:, ...], gen_t2_ik)
            loss_ki = self.criterion(img_gt[:, :2, ...], gen_t2_ki)
            loss_total = loss_i + loss_k / self.k_weight + self.alpha * (loss_ik / self.k_weight + loss_ki)
            return [loss_total, loss_i, loss_k, loss_ik, loss_ki]
        else:
            gen_t2 = self.pred(img_input, u_mask, img_mask)
            loss_total = self.criterion(img_gt, gen_t2)
            return [loss_total]

    def get_training_pairs(self, imgs_t1, imgs_t2, imgs_t2u, u_mask, img_masks):
        if self.task == 'trans':
            model_input = imgs_t1
            model_gt = imgs_t2
        elif self.task == 'reg':
            trans_t2 = self.trans_module.pred(imgs_t1, u_mask, img_masks[0])
            if self.domain == 2:
                trans_t2 = torch.cat((trans_t2[0], trans_t2[1]), 1)
            model_input = torch.cat((trans_t2, imgs_t2u), 1)
            model_gt = imgs_t2
        elif self.task == 'recon':
            if self.recon_modal == 't2u':
                model_input = imgs_t2u
                model_gt = imgs_t2
            elif self.recon_modal == 't1t2u':
                model_input = torch.cat((imgs_t1, imgs_t2u), 1)
                model_gt = imgs_t2
            else:
                if self.trans_module is None:
                    assert self.reg_module is None
                    trans_t2 = imgs_t1
                # T2s
                else:
                    trans_t2 = self.trans_module.pred(imgs_t1, u_mask, img_masks[0])
                    if self.domain == 2:
                        trans_t2 = torch.cat((trans_t2[0], trans_t2[1]), 1)

                # T1/T2s + T2u
                reg_input = torch.cat((trans_t2, imgs_t2u), 1)

                # T2s
                if self.reg_module is None:
                    if self.domain == 2:
                        warped_t2 = reg_input[:, :4]
                    else:
                        warped_t2 = reg_input[:, :2]
                # T2sw
                else:
                    warped_t2 = self.reg_module.pred(reg_input, u_mask, img_masks[1])
                    if self.domain == 2:
                        warped_t2 = torch.cat((warped_t2[0], warped_t2[1]), 1)

                model_input = torch.cat((warped_t2, imgs_t2u), 1)
                model_gt = imgs_t2

        return (model_input, model_gt)

    def test(self, dataloader, epoch):
        u_mask = cartesian_mask((1, dataloader.dataset.dim[0], dataloader.dataset.dim[1]), self.un_rate)[0,]
        u_mask = Tensor(u_mask)

        self.load_model_weight(epoch)

        if is_dudo(self.domain):
            self.model[0].eval()
            self.model[1].eval()
        else:
            self.model[0].eval()

        t2u_psnr_list = []
        recon_psnr_list = []
        recon_ssim_list = []

        if self.domain == 2:
            recon_psnr_list_k = []
            recon_ssim_list_k = []

        for i, imgs in enumerate(dataloader):

            imgs_t1, imgs_t2, imgs_t2u, t1_mask, t2_mask = get_data_pair(imgs, u_mask, self.domain)

            model_input, model_gt = self.get_training_pairs(imgs_t1, imgs_t2, imgs_t2u, u_mask, (t1_mask, t2_mask))

            if self.domain == 2:
                if self.task == 'trans':
                    img_input = imgs_t1[:, :2]
                    k_input = imgs_t1[:, 2:]
                elif self.task in ['reg', 'recon']:
                    img_input = torch.concat((model_input[:, :2], model_input[:, 4:6]), 1)
                    k_input = torch.concat((model_input[:, 2:4], model_input[:, 6:]), 1)

                if self.task == 'reg':
                    pred_t2_i = pred_t2_img(self.model[0], img_input, self.task, t2_mask, 1, u_mask)
                    pred_t2_k = pred_t2_img(self.model[0], k_input, self.task, t2_mask, 0, u_mask)
                else:
                    pred_t2_i = pred_t2_img(self.model[0], img_input, self.task,
                                            t1_mask if self.task == 'trans' else t2_mask, 1, u_mask)
                    pred_t2_k = pred_t2_img(self.model[1], k_input, self.task,
                                            t1_mask if self.task == 'trans' else t2_mask, 0, u_mask)

            else:
                pred_t2 = pred_t2_img(self.model[0], model_input, self.task,
                                      t1_mask if self.task == 'trans' else t2_mask, self.domain, u_mask)

            if self.domain == 0:
                pred_t2_un = torch.complex(pred_t2[:, :1], pred_t2[:, 1:])
                imgs_t2_un = torch.complex(imgs_t2[:, :1], imgs_t2[:, 1:])
                imgs_t2u_un = torch.complex(imgs_t2u[:, :1], imgs_t2u[:, 1:])
                pred_t2 = c_k2r_i(pred_t2_un)
                imgs_t2 = c_k2r_i(imgs_t2_un)
                imgs_t2u = c_k2r_i(imgs_t2u_un)
            elif self.domain == 2:
                pred_t2_k_un = torch.complex(pred_t2_k[:, :1], pred_t2_k[:, 1:])
                pred_t2_ki = c_k2r_i(pred_t2_k_un)
            # Whole brain
            if self.domain == 1:
                t2u_psnr, pred_psnr, pred_ssim = get_psnr_ssim(imgs_t2, pred_t2, imgs_t2u)
            elif self.domain == 0:
                t2u_psnr, pred_psnr, pred_ssim = get_psnr_ssim(imgs_t2[:, :2], pred_t2[:, :2], imgs_t2u)
            elif self.domain == 2:
                t2u_psnr, pred_psnr, pred_ssim = get_psnr_ssim(imgs_t2[:, :2], pred_t2_i, imgs_t2u[:, :2])
                _, pred_psnr_k, pred_ssim_k = get_psnr_ssim(imgs_t2[:, :2], pred_t2_ki,
                                                            imgs_t2u[:, :2])
                recon_psnr_list_k.append(pred_psnr_k)
                recon_ssim_list_k.append(pred_ssim_k)
            t2u_psnr_list.append(t2u_psnr)
            recon_psnr_list.append(pred_psnr)
            recon_ssim_list.append(pred_ssim)


        print('Best PSNR for the current dataset: ' + str(np.mean(recon_psnr_list)) + ' ' + str(np.std(recon_psnr_list)))
        print('Best SSIM for the current dataset: ' + str(np.mean(recon_ssim_list)) + ' ' + str(np.std(recon_ssim_list)))
        if is_dudo(self.domain):
            print('Best PSNR for the current dataset (k): ' + str(np.mean(recon_psnr_list_k)) + ' ' + str(np.std(recon_psnr_list_k)))
            print('Best SSIM for the current dataset (k): ' + str(np.mean(recon_ssim_list_k)) + ' ' + str(np.std(recon_ssim_list_k)))

        self.load_model_weight(epoch)


    def write_summary(self, writer, step, losses):
        writer.add_scalar('loss', losses[0].item(), global_step=step)
        if is_dudo(self.domain):
            writer.add_scalar('loss_i', losses[1].item(), global_step=step)
            writer.add_scalar('loss_k', losses[2].item(), global_step=step)
            writer.add_scalar('loss_ik', losses[3].item(), global_step=step)
            writer.add_scalar('loss_ki', losses[4].item(), global_step=step)


def is_dudo(domain):
    return domain == 2
