import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from scipy.spatial.distance import directed_hausdorff


class HD95(nn.Module):
    def __init__(self):
        super(HD95, self).__init__()

    def forward(self, mask1, mask2):
        assert mask1.shape == mask2.shape, "the sizes of masks must be the same"
        assert mask1.dtype == torch.int64 and mask2.dtype == torch.int64, "the dtype of masks should be torch.int64"

        batch_size, _, height, width = mask1.shape
        hd95_values = []

        for i in range(batch_size):
            mask1_i = mask1[i, 0].cpu().numpy()
            mask2_i = mask2[i, 0].cpu().numpy()

            unique_labels = np.unique(np.concatenate((mask1_i, mask2_i)))
            unique_labels = unique_labels[unique_labels != 0] 

            hd95_for_all_classes = []

            for label in unique_labels:
                mask1_coords = np.argwhere(mask1_i == label)
                mask2_coords = np.argwhere(mask2_i == label)

                if len(mask1_coords) == 0 or len(mask2_coords) == 0:
                    continue

                h1 = directed_hausdorff(mask1_coords, mask2_coords)[0]
                h2 = directed_hausdorff(mask2_coords, mask1_coords)[0]

                h1_array = np.array([h1])
                h2_array = np.array([h2])

                hd95 = np.percentile(np.concatenate([h1_array, h2_array]), 95)
                hd95_for_all_classes.append(hd95)

            if len(hd95_for_all_classes) == 0:
                hd95_values.append(np.inf)
            else:
                hd95_values.append(np.mean(hd95_for_all_classes))

        return torch.tensor(hd95_values, device=mask1.device).mean()

class Jacobian(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, deformation_field):
        dx = deformation_field[:, 0]  # [B,H,W]
        dy = deformation_field[:, 1]  # [B,H,W]

        grad_dx = torch.gradient(dx, dim=[-2, -1])  # (dy_dx, dx_dx)
        grad_dy = torch.gradient(dy, dim=[-2, -1])  # (dy_dy, dx_dy)

        dx_dx = grad_dx[1]   
        dx_dy = grad_dx[0]  
        dy_dx = grad_dy[1]   
        dy_dy = grad_dy[0]  

        det_jacobian = dx_dx * dy_dy - dx_dy * dy_dx


        negative_mask = (det_jacobian <= 0)
        negative_ratio = torch.mean(negative_mask.float()) * 100  

        return negative_ratio


class Dice(nn.Module):
    def __init__(self, smooth=1e-5, ignore_bg=True):
        super(Dice, self).__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, mask1, mask2):
        assert mask1.shape == mask2.shape, "the sizes of masks must be the same"
        assert mask1.dtype == torch.int64 and mask2.dtype == torch.int64, "the dtype of masks should be torch.int64"

        num_classes = 3  
        mask1_one_hot = self._one_hot(mask1, num_classes)
        mask2_one_hot = self._one_hot(mask2, num_classes)

        dice_values = []
        for i in range(num_classes):
            dice_i = self._dice(mask1_one_hot[:, i], mask2_one_hot[:, i])
            dice_values.append(dice_i)

        dice_values = torch.stack(dice_values, dim=1) 

        if self.ignore_bg:
            return dice_values[:,1:].mean()
        else:
            return dice_values.mean()
        
    def _one_hot(self, mask, num_classes):
        mask = mask.squeeze(1)  
        one_hot = torch.zeros(mask.shape[0], num_classes, mask.shape[1], mask.shape[2], device=mask.device)
        return one_hot.scatter_(1, mask.unsqueeze(1), 1)

    def _dice(self, mask1, mask2):

        mask1 = mask1.float()
        mask2 = mask2.float()

        intersection = torch.sum(mask1 * mask2, dim=(1, 2))
        sum_mask1 = torch.sum(mask1, dim=(1, 2))
        sum_mask2 = torch.sum(mask2, dim=(1, 2))

        dice = (2.0 * intersection + self.smooth) / (sum_mask1 + sum_mask2 + self.smooth)
        return dice


class MSE(nn.Module):
    """
    Mean squared error loss.
    """

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class SSIM_loss(torch.nn.Module):
    def __init__(self, if_MS=True, win_size=11):
        super(SSIM_loss, self).__init__()
        if if_MS:
            self.SSIM = MS_SSIM(win_size=win_size, data_range=255, size_average=True, channel=1)
        else:
            self.SSIM = SSIM(data_range=255, size_average=True, channel=1)
    def forward(self, img1, img2):
        return 1-self.SSIM(img1, img2)


class DiscrepancyRate(nn.Module):
    def __init__(self):
        super(DiscrepancyRate,self).__init__()
    
    def forward(self,def_map,force):
        """
        def_map: deformation field of shape[B,2,H,W] [x,y]
        force: force data of shape [B,2] [moving,fixed]
        """
        df = force[:,1]-force[:,0] 
        count = (df.unsqueeze(1).unsqueeze(2) * def_map[:,0,:,:]) > 0 
        dcy_rate = 1 - count.sum()/def_map[:,0,:,:].numel()
        return dcy_rate


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        # print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

