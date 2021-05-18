import torch
from torch.nn.functional import mse_loss

from torchvision.transforms import ToTensor

def psnr(sr_img, hr_img):
    """
    Method to calculate the PSNR loss given
    a SR and a HR image 
    @param sr_img:Tensor Generated image
    @param hr_img:Tensor Ground truth

    @returns int
    """
    mse = mse_loss(sr_img, hr_img, size_average=None, reduce=None, reduction='mean')
    max_pixel_value = torch.tensor(255)
    psnr = 20*torch.log10(max_pixel_value)-10*torch.log10(mse)
    return psnr.item()
