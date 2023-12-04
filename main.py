import torch
from torch.utils.data import DataLoader

from datasets import BraTS2019Full, BraTSSubset, k_fold_split

import argparse

from module import ModelModule

plane_to_dim = {'saggital': 0, 'coronal': 1, 'axial': 2}
domain_to_str = {0: 'k', 1: 'img', 2: 'dual'}

if __name__ == "__main__":
    print('start')
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=int, default=2, help='domain to be trained using, 0 for k-space, 1 for image, 2 for dual domain.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--k_weight', type=float, default=100)
    parser.add_argument('--qmodal', type=str, default='T2')
    parser.add_argument('--qm_only', type=bool, default=False)
    parser.add_argument('--un_rate', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default='BraTS')
    parser.add_argument('--dataset_path', type=str, default='./MICCAI_BraTS_2019_Data_Training')
    parser.add_argument('--recon_only', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./saved_models/')
    parser.add_argument('--network', type=str, default='unet')
    parser.add_argument('--plane', type=str, default='axial')
    parser.add_argument('--num_consecutive', type=int, default=1, help='number of consecutive to extract from eachvolume.')
    parser.add_argument('--train_epoch', type=int, default=100)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    if args.dataset_name == 'BraTS':
        DIM = BraTS2019Full(args.qmodal, dim=plane_to_dim[args.plane]).crop_size
        full_filelist = BraTS2019Full(args.qmodal, dim=plane_to_dim[args.plane]).file_list
    else:
        raise NotImplementedError

    trans_name = '_'.join(
        [args.dataset_name, 'trans', domain_to_str[args.domain], 'un' + str(args.un_rate), args.plane,
         args.qmodal])

    reg_name = '_'.join(
        [args.dataset_name, 'reg', domain_to_str[args.domain], 'un' + str(args.un_rate), args.plane,
         args.qmodal])

    recon_name = '_'.join(
        [args.dataset_name, 'recon', domain_to_str[args.domain], 'un' + str(args.un_rate), args.plane,
         args.qmodal])

    if args.dataset_name == 'BraTS':
        train_list, validation_list, test_list = k_fold_split(1, full_filelist, 0)
        ts = BraTSSubset(train_list, args.qmodal, DIM, num_consecutive=args.num_consecutive, aug=args.aug,
                         domain=args.domain)
        dataloader_train = DataLoader(
            ts,
            batch_size=args.batch_size,
            shuffle=True
        )

        dataloader_val = DataLoader(
            BraTSSubset(validation_list, args.qmodal, DIM, num_consecutive=args.num_consecutive,
                        aug=args.aug, domain=args.domain),
            batch_size=args.batch_size,
        )

        dataloader_test = DataLoader(
            BraTSSubset(test_list, args.qmodal, DIM, num_consecutive=args.num_consecutive, aug=args.aug,
                        domain=args.domain),
            batch_size=1,
        )
    else:
        raise NotImplementedError

    if args.recon_only:
        # recon only with t2u or t1 + t2u
        recon_module = ModelModule(args.domain, recon_name, 'recon', args.un_rate,
                                   recon_modal=args.recon_modal, alpha=args.alpha, k_weight=args.k_weight)
        recon_module.train(dataloader_train, args.train_epoch)
        recon_module.test(dataloader_test, [args.train_epoch])

    else:
        trans_module = ModelModule(args.domain, trans_name, 'trans', args.un_rate, trained_dataloader=dataloader_train, alpha=args.alpha, k_weight=args.k_weight)
        trans_module.train(dataloader_train, args.train_epoch)
        trans_module = trans_module.test(dataloader_test, [args.train_epoch])

        reg_module = ModelModule(args.domain, reg_name, 'reg', args.un_rate, trans_module=trans_module, alpha=args.alpha, k_weight=args.k_weight)
        reg_module.train(dataloader_train, args.train_epoch)
        reg_module = reg_module.test(dataloader_test, [args.train_epoch])

        recon_module = ModelModule(args.domain, recon_name, 'recon', args.un_rate, trans_module=trans_module, reg_module=reg_module, alpha=args.alpha, k_weight=args.k_weight)
        recon_module.train(dataloader_train, args.train_epoch)
        recon_module.test(dataloader_test, [args.train_epoch])

    recon_module.test(dataloader_test, [args.train_epoch])

