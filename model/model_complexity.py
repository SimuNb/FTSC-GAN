from model.ftsc_gan import FTSC_GAN
import torch
import numpy as np
from ptflops import get_model_complexity_info

def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num




with torch.cuda.device(0):
    net = FTSC_GAN()
    print(numParams(net))
    macs, params = get_model_complexity_info(net, (1, 16384), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))