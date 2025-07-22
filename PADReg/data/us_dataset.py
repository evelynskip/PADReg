import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_pair_list(filename, delim=None, prefix=None, suffix=None):
    '''
    Reads a list of registration file pairs from a line-seperated text file.

    Parameters:
        filename: Filename to load.
        delim: File pair delimiter. Default is a whitespace seperator (None).
        prefix: File prefix. Default is None.
        suffix: File suffix. Default is None.
    '''
    with open(filename, 'r') as file:
        content = file.readlines()
    linelist = [x.strip() for x in content if x.strip()]
    pairlist = [f.split(delim) for f in linelist]
    if prefix is not None:
        pairlist = [[prefix + f for f in pair] for pair in pairlist]
    if suffix is not None:
        pairlist = [[f + suffix for f in pair] for pair in pairlist]
    return pairlist

def load_image(filename):
    """
    Parameters:
        filename: path of the image file
    Return:
        img: ndarray [C,H,W]
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)
    if filename.endswith('.png'):
        image = cv2.imread(filename,0)
        image_np = image[None,...]
        image_np = image_np.astype(np.float32)
        image_np /= 255
        # image_torch = torch.from_numpy(image_np)
        return image_np
    else:
        raise ValueError("The file type is not supported.")
    
def load_force(pairname,
    force_folder
):
    """
    Return:

    """
    video_id = pairname[0][-23:-8]
    moving_fid = int(pairname[0][-7:-4])
    fixed_fid = int(pairname[1][-7:-4])
    frc_file = os.path.join(force_folder,video_id+'.txt')
    frc_data = np.loadtxt(frc_file,delimiter=' ')
    forces = torch.tensor([frc_data[moving_fid-1,2],frc_data[fixed_fid-1,2]]).float()
    return -forces #TODO

def load_annotations(pairname,
    anno_folder,
    size=(256,256) #TODO
):
    moving_fid = pairname[0][-23:]
    fixed_fid = pairname[1][-23:]
    m_ann_path = os.path.join(anno_folder,moving_fid)
    f_ann_path = os.path.join(anno_folder,fixed_fid)
    m_ann_ = cv2.imread(m_ann_path,0)
    f_ann_ = cv2.imread(f_ann_path,0)
    # Resize mask
    m_ann_ = cv2.resize(m_ann_, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    f_ann_ = cv2.resize(f_ann_, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    m_ann = torch.from_numpy(m_ann_[None,...]).long()
    f_ann = torch.from_numpy(f_ann_[None,...]).long()
    return m_ann, f_ann 

class USDataset(Dataset):
    def __init__(self, data_list, prefix, force_folder=None, transforms=None, ann_folder=None):
        # get image and force list 
        self.paths = read_pair_list(data_list, prefix=prefix)
        self.force_folder = force_folder
        self.ann_folder = ann_folder
        # get transforms
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Return:
            moving_img: torch tensor [B,C=1,H,W]
            fixed_img: torch tensor ndarray [B,C=1,H,W]
            force: [B,2] > 0 
            m_ann:[B,C=1,H,W]
            f_ann:[B,C=1,H,W]
        """
        # get pair_name: list [moving_filename, fixed_filename]
        pair_name = self.paths[index]

        # get moving and fixed image
        moving_img = load_image(pair_name[0])
        fixed_img = load_image(pair_name[1])
        if self.transforms:
            moving_img,fixed_img = self.transforms([moving_img, fixed_img])
        
        moving_img, fixed_img = torch.from_numpy(moving_img), torch.from_numpy(fixed_img)
        output = [moving_img,fixed_img]

        # Get according force data
        if self.force_folder:
            force = load_force(pair_name,self.force_folder)
            output.append(force)
        # Get according annotations
        if self.ann_folder:
            m_ann, f_ann = load_annotations(pair_name,self.ann_folder)
            output.append(m_ann)
            output.append(f_ann)
        return output
    
    def __len__(self):
        return len(self.paths)


class USInferDataset(Dataset):
    def __init__(self, data_list, prefix, force_folder=None, transforms=None, ann_folder=None):
        # get image and force list 
        self.paths = read_pair_list(data_list, prefix=prefix)
        self.force_folder = force_folder
        self.ann_folder = ann_folder
        # get transforms
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Return:
            moving_img: torch tensor [B,C=1,H,W]
            fixed_img: torch tensor ndarray [B,C=1,H,W]
            force: [B,2,L=6]
        """
        # get pair_name: list [moving_filename, fixed_filename]
        pair_name = self.paths[index]

        # get moving and fixed image
        moving_img = load_image(pair_name[0])
        fixed_img = load_image(pair_name[1])
        if self.transforms:
            moving_img,fixed_img = self.transforms([moving_img, fixed_img])
        moving_img, fixed_img = torch.from_numpy(moving_img), torch.from_numpy(fixed_img)
        moving_img_rgb = moving_img.repeat(3,1,1)#.permute(1,2,0)
        fixed_img_rgb = fixed_img.repeat(3,1,1)#.permute(1,2,0)
        output = [moving_img_rgb, fixed_img_rgb, moving_img, fixed_img]

        # get according force data
        if self.force_folder:
            force = load_force(pair_name,self.force_folder)
            output.append(force)
        # Get according annotations
        if self.ann_folder:
            m_ann, f_ann = load_annotations(pair_name,self.ann_folder)
            output.append(m_ann)
            output.append(f_ann)
        return output

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __len__(self):
        return len(self.paths)