import torch
from torch import nn


def count_parameters(model) -> int:
    """ Counts the number of learnable parameters of a given model"""
    return sum(p.numel() for p in model.parameters())    


class Conv2dTo3d(nn.Module):
    def __init__(self, input_module, kernel_size = None):
        super(Conv2dTo3d, self).__init__()

        self.in_channels = input_module.in_channels
        self.out_channels = input_module.out_channels
        # for the rest, assume kernel is symmetric
        self.kernel_size = input_module.kernel_size[0] 
        self.stride = input_module.stride[0]
        self.padding = input_module.padding[0]
        self.transposed = isinstance(input_module, nn.ConvTranspose2d)

        bias = input_module.bias
        if bias is not None:
            self.bias = nn.Parameter(bias) # finetune the already optimized bias
        else:
            self.bias = bias

        if self.transposed:
            self.weight_3d = nn.Parameter(
                torch.zeros(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, self.kernel_size)
            )
            self.weight_3d.data[:] = 1e-8
            with torch.no_grad():
                if self.kernel_size % 2 == 1:
                    self.weight_3d[:,:,self.kernel_size//2 + 1] = input_module.weight.data.clone()
                else:
                    self.weight_3d[:] = input_module.weight.data[:,:,None].clone() / self.kernel_size
       
        else:
            self.weight_3d = nn.Parameter(
                torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
            )
            self.weight_3d.data[:] = 1e-8
            with torch.no_grad():
                if self.kernel_size % 2 == 1:
                    self.weight_3d[:,:,self.kernel_size//2 + 1] = input_module.weight.data.clone()
                else:
                    self.weight_3d[:] = input_module.weight.data[:,:,None].clone() / self.kernel_size
            
        
    def forward(self, x):
        weight_3d = self.weight_3d
        scaling = 1.0
            
        if self.transposed:
            output = nn.functional.conv_transpose3d(
                x, weight_3d, bias=self.bias, stride=self.stride, padding=self.padding
            ) * scaling

        else:
            output = nn.functional.conv3d(
                x, weight_3d, bias=self.bias, stride=self.stride, padding=self.padding
            ) * scaling

        return output


def transform_2d_to_3d(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            conv3d = Conv2dTo3d(module)
            setattr(model, name, conv3d)
        else:
            transform_2d_to_3d(module)
