
import torch.nn as nn
import torch.nn.init as init
import torch

class Stiff2Flow(nn.Module):
    def __init__(self):
        super(Stiff2Flow,self).__init__()
        """flow estimation using force"""
        self.x10 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.x11 = nn.Parameter(torch.randn(1, requires_grad=True))
    def forward(self, stiff, force):
        """
        Parameters:
            force[B 2](moving, fixed) stiff[B 2 H W]
        Return:
            flow[B 2 H W]    
        """
        force_d = (force[:,1] - force[:,0]) / (force[:,0] + force[:,1])
        force_d_abs = torch.abs(force_d)
        force_d_sqrt = torch.sqrt(force_d_abs)
        force_d = torch.where(force_d >= 0,force_d_sqrt, -force_d_sqrt)
       
        force_d = force_d[...,None,None,None]
        stiff_x, stiff_y = stiff[:,[0],:,:], stiff[:,[1],:,:]

        """Calculate deformation field using force"""
        dx = force_d * stiff_x
        dy = force_d * stiff_y

        """ Concate dx and dy """
        flow = torch.cat((dx, dy), dim=1)
        return flow

