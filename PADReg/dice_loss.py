""" calculation of DICE

Typical usage example:

diceitem = dice_coeff(pred, true_masks).item()
"""
import torch
from torch.autograd import Function
import torch.nn.functional as F

class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    def forward(self, input, target):
        input = F.one_hot(input.squeeze(1), num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
        target = F.one_hot(target.squeeze(1), num_classes=3).permute(0, 3, 1, 2).to(torch.float32)
        self.save_for_backward(input, target)
        eps = 1e-3
        
        # input = input[1:,:,:]
        # target = target[1:,:,:]
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        if t>1:
            print(self.inter,torch.sum(input), torch.sum(target))
            # print(torch.min(input), torch.min(input))
            # print(input.shape, target.shape)
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, target = self.saved_variables
        grad_input = grad_target = None
        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None
        return grad_input, grad_target

"""Dice coeff for batches

    Calculate the Dice coeff for batches

    Args:
        input: mask result [B,1,H,W] type: long
        target: label [B,1,H,W] type: long

    Returns:
        DICE: %
    """
def dice_coeff(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def dice_loss(input, target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])
    return 1- s / (i + 1)