import glob
import os
import numpy as np

import torch
from torch.utils.data import Dataset

import nibabel as nib

from scipy.ndimage.interpolation import rotate

from signal_utils import fft

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


class BraTS2019Full(Dataset):
    def __init__(self, query_modal, dataset_path='./MICCAI_BraTS_2019_Data_Training/',
                 volume_size=(240, 240, 155), n_slices=16, dim=2):
        self.query_modal = query_modal
        self.crop_size = (192, 192, 128)[:dim] + (192, 192, 128)[dim+1:]
        self.root = str(dataset_path)
        self.full_vol_dim = volume_size
        self.file_list = []
        self.dim = dim
        # Define path and file names to save
        subvol = '_vol_' + str(volume_size[0]) + 'x' + str(volume_size[1]) + 'x' + str(volume_size[2]) + 'x' + str(
            n_slices) + '_' + str(dim)
        self.sub_vol_path = self.root + 'generated/' + subvol + '/'
        make_dirs(self.sub_vol_path)

        # Check if data already generated
        if not already_generated(self.sub_vol_path):
            # If data not generated, save data as numpy array
            list_IDsT1 = sorted(glob.glob(os.path.join(self.root, '*GG/*/*t1.nii.gz')))
            list_IDsT2 = sorted(glob.glob(os.path.join(self.root, '*GG/*/*t2.nii.gz')))
            list_IDsF = sorted(glob.glob(os.path.join(self.root, '*GG/*/*flair.nii.gz')))
            labels = sorted(glob.glob(os.path.join(self.root, '*GG/*/*_seg.nii.gz')))

            print('Brats2019, Total data:', len(list_IDsT1))

            save_local_volumes(list_IDsT1, list_IDsT2, list_IDsF, labels, n_channels=n_slices,
                                        sub_vol_path=self.sub_vol_path, crop_size=self.crop_size, dim=self.dim)
        self.file_list = extract_list(self.sub_vol_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        f_flair, f_t1, f_t2, f_label = np.array(self.file_list[index]).T
        img_t1 = np.vstack([np.load(t1) for t1 in f_t1])
        img_t2 = np.vstack([np.load(t2) for t2 in f_t2])
        img_flair = np.vstack([np.load(flair) for flair in f_flair])
        img_label = np.vstack([np.load(label) for label in f_label])
        img_query = get_query_img(self.query_modal, img_t2, img_flair)
        return img_t1, img_query, img_label


class BraTSSubset(Dataset):
    def __init__(self, file_list, query_modal, dim, num_consecutive=3, aug=False, domain=-1):
        self.file_list = file_list
        self.num_consecutive = num_consecutive
        self.query_modal = query_modal
        self.aug = aug
        self.domain = domain
        self.dim = dim


    def __len__(self):
        return len(self.file_list) - self.num_consecutive + 1

    def __getitem__(self, index):
        f_flair, f_t1, f_t2, f_label = np.array(self.file_list[index:index + self.num_consecutive]).T
        img_t1 = np.vstack([np.load(t1) for t1 in f_t1])
        if self.query_modal == 'T2':
            img_query = np.vstack([np.load(t2) for t2 in f_t2])
        elif self.query_modal == 'FLAIR':
            img_query = np.vstack([np.load(flair) for flair in f_flair])
        else:
            raise NotImplementedError
        img_label = np.vstack([np.load(label) for label in f_label])
        if self.aug:
            img_t1, img_query = augment_data(img_t1[..., np.newaxis], img_query[..., np.newaxis])
            img_t1, img_query = img_t1[...,0], img_query[0,...,0][np.newaxis,...]

        t1_mask = (img_t1 != 0)
        query_mask = (img_query != 0)

        if self.domain == 0: # k
            img_t1 = fft(img_t1)
            img_t1 = np.vstack([img_t1.real, img_t1.imag])
            # real and imaginary
            img_query = fft(img_query)
            img_query = np.vstack([img_query.real, img_query.imag])
        elif self.domain == 1: # image
            img_t1 = np.vstack([img_t1, np.zeros_like(img_t1)])
            img_query = np.vstack([img_query, np.zeros_like(img_query)])
        elif self.domain == 2: # dual
            img_t1_k = fft(img_t1)
            img_query_k = fft(img_query)
            img_t1 = np.vstack([img_t1, np.zeros_like(img_t1), img_t1_k.real, img_t1_k.imag])
            img_query = np.vstack([img_query, np.zeros_like(img_query), img_query_k.real, img_query_k.imag])
        return img_t1, img_query, img_label, (t1_mask, query_mask)


def load_med_image(path, dtype):
    # Load data from path as array
    img_nii = nib.load(path)
    img_np = np.squeeze(img_nii.get_fdata(dtype=np.float32))

    # Normalize
    if dtype != 'label':
        data_max, data_min = np.max(img_np), np.min(img_np)
        img_np = (img_np - data_min) / (data_max - data_min + 1e-5)

    return img_np


def extract_volumes(path, img_np, modality, n_channels, crop_size, dim, rand_range=None):
    paths = []
    w, h, d = img_np.shape
    x, y, z = np.where(img_np != 0)

    if dim == 0:
        if rand_range is None:
            rand_range = np.random.choice(range(min(x), max(x)), n_channels, replace=False)
        modified_np = img_np[rand_range, :, :]
    elif dim == 1:
        if rand_range is None:
            rand_range = np.random.choice(range(min(y), max(y)), n_channels, replace=False)
        modified_np = img_np[:, rand_range, :]
        modified_np = np.transpose(modified_np, (1, 0, 2))
    elif dim == 2:
        if rand_range is None:
            rand_range = np.random.choice(range(min(z), max(z)), n_channels, replace=False)
        modified_np = img_np[:, :, rand_range]
        modified_np = np.transpose(modified_np, (2, 0, 1))
    else:
        raise NotImplementedError

    crop_1, crop_2 = crop_size

    for i, j in enumerate(rand_range):
        curr_fname = path + str(j) + '_' + modality + '.npy'
        if i >= modified_np.shape[0]:
            break
        to_be_saved = modified_np[i,][np.newaxis,]
        if dim == 0:
            to_be_saved = to_be_saved[:, int(h / 2 - crop_1 / 2):int(h / 2 + crop_1 / 2),
                          int(d / 2 - crop_2 / 2):int(d / 2 + crop_2 / 2)]
        elif dim == 1:
            to_be_saved = to_be_saved[:, int(w / 2 - crop_1 / 2):int(w / 2 + crop_1 / 2),
                          int(d / 2 - crop_2 / 2):int(d / 2 + crop_2 / 2)]
        elif dim == 2:
            to_be_saved = to_be_saved[:, int(w / 2 - crop_1 / 2):int(w / 2 + crop_1 / 2),
                          int(h / 2 - crop_2 / 2):int(h / 2 + crop_2 / 2)]
        np.save(curr_fname, to_be_saved)
        paths.append(curr_fname)
    return paths, rand_range


def make_dirs(gen_path):
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)


def already_generated(gen_path):
    return len(os.listdir(gen_path)) > 0


def extract_list(gen_path):
    fnames = sorted(os.listdir(gen_path))
    lst = []
    for i in range(0, len(fnames), 4):
        lst.append((gen_path + fnames[i], gen_path + fnames[i + 1], gen_path + fnames[i + 2], gen_path + fnames[i + 3]))
    return lst


def idx2modality(x):
    if x == 0:
        return 'T1'
    elif x == 1:
        return 'T2'
    elif x == 2:
        return 'FLAIR'
    elif x == 3:
        return 'label'
    else:
        raise NotImplementedError


def save_local_volumes(*ls, n_channels, sub_vol_path, crop_size, dim):
    # Prepare variables
    total = len(ls[0])
    assert total != 0, "Problem reading data. Check the data paths."
    modalities = len(ls)
    path_lst = []

    print("Volumes: ", total)
    for i in range(total):
        filename = sub_vol_path + 'id_' + str(i) + '_batch_'
        list_saved_paths = []

        for j in range(modalities):
            t = 'label' if j == 3 else 'T1/T2/F'
            img_np = load_med_image(ls[j][i], t)
            if j == 0:
                split_fnames, rand_range = extract_volumes(filename, img_np, idx2modality(j), n_channels, crop_size, dim=dim)
            else:
                split_fnames, _ = extract_volumes(filename, img_np, idx2modality(j), n_channels, crop_size, dim=dim, rand_range=rand_range)
            list_saved_paths.append(split_fnames)
        list_saved_paths = [(list_saved_paths[0][k], list_saved_paths[1][k], list_saved_paths[2][k], list_saved_paths[3][k]) for k in
                            range(len(list_saved_paths[0]))]
        path_lst += list_saved_paths
    return path_lst


def k_fold_split(n_folds, file_list, curr_fold):
    assert curr_fold < n_folds
    get_id = lambda x: int(x[0].split('_')[9])
    num_ids = len(list(set([get_id(i) for i in file_list])))
    each_fold_len = int(num_ids / n_folds)
    test_fold = [i for i in file_list if get_id(i) in list(range(each_fold_len * curr_fold, each_fold_len * (curr_fold + 1)))]
    if curr_fold == n_folds - 1:
        validation_fold = [i for i in file_list if get_id(i) in list(range(each_fold_len))]
    else:
        validation_fold = [i for i in file_list if get_id(i) in list(range(each_fold_len * (curr_fold + 1), each_fold_len * (curr_fold + 2)))]
    train_fold = [i for i in file_list if i not in validation_fold + test_fold]
    return train_fold, validation_fold, test_fold



def get_query_img(query_modal, img_t2, img_flair):
    if query_modal == 't2':
        img_query = img_t2
    elif query_modal == 'flair':
        img_query = img_flair
    else:
        raise ValueError('Query modality should be T2 or FLAIR.')
    return img_query


def augment_data(t1_gt, query_gt):
    # dim: [nb, nx, ny, nc]
    assert t1_gt.ndim == 4 and query_gt.ndim == 4
    np.random.seed(777)
    # randomly rotate t1
    rotated = random_rotate(t1_gt)
    # random translate t1
    translated = random_translate(rotated)
    return translated, query_gt


def random_rotate(img, rotate_range=(-5, 5)):
    _, h, w, _ = img.shape
    rotated_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        angle = np.random.randint(*rotate_range)
        tmp = rotate(img[i], angle, reshape=False, order=0)
        tmp = tmp[tmp.shape[0]//2-h//2:tmp.shape[0]//2+h//2, tmp.shape[1]//2-w//2:tmp.shape[1]//2+w//2,:]
        rotated_img[i] = tmp
    return rotated_img


def random_translate(img, translate_range=(-5, 5)):
    _, h, w, _ = img.shape
    translated_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        interval_h = np.random.randint(*translate_range)
        interval_w = np.random.randint(*translate_range)
        tmp_img = np.zeros([h+2*abs(interval_h), w+2*abs(interval_w), 1])
        tmp_img[tmp_img.shape[0]//2-h//2+interval_h:tmp_img.shape[0]//2+h//2+interval_h,tmp_img.shape[1]//2-w//2+interval_w:tmp_img.shape[1]//2+w//2+interval_w] = img[i]
        translated_img[i] = tmp_img[tmp_img.shape[0]//2-h//2:tmp_img.shape[0]//2+h//2,tmp_img.shape[1]//2-w//2:tmp_img.shape[1]//2+w//2]
    return translated_img

