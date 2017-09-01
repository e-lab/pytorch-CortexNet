import torch
from torch import nn
from torch.nn import functional as f
from torch.autograd import Variable as V

from collections import OrderedDict

class EncDec(nn.Module):
    '''
    A slightly modified version of cortexnet model02.
    No classifier attached. 
    '''
    KERNEL_SIZE = 3
    PADDING = KERNEL_SIZE // 2
    KERNEL_STRIDE = 2

    def __init__(self, network_size):
        super().__init__()
        self.nlayers = len(network_size) - 1
        
        for layer in range(self.nlayers):
            # Create D[layer] block
            D = nn.Conv2d(in_channels = network_size[layer],
                          out_channels = network_size[layer + 1],
                          kernel_size = CortexNetBase.KERNEL_SIZE,
                          stride = CortexNetBase.KERNEL_STRIDE,
                          padding = CortexNetBase.PADDING)
            D_BN = nn.BatchNorm2d(network_size[layer+1])

            # Create G[layer] block
            G = nn.ConvTranspose2d(in_channels = network_size[layer+1],
                                   out_channels = network_size[layer],
                                   kernel_size = CortexNetBase.KERNEL_SIZE,
                                   stride = CortexNetBase.KERNEL_STRIDE,
                                   padding = CortexNetBase.PADDING,
                                   output_padding=CortexNetBase.PADDING)
            G_BN = nn.BatchNorm2d(network_size[layer])

            setattr(self, 'D_'+str(layer+1), D)
            setattr(self, 'D_'+str(layer+1)+'_BN', D_BN)
            setattr(self, 'G_'+str(layer+1), G)
            if layer > 0 : setattr(self, 'G_'+str(layer+1)+'_BN', G_BN)

    def forward(self, x, all_layers = False):

        residuals = []
        outputs = OrderedDict()

        for layer in range(self.nlayers):
            D = getattr(self, 'D_'+str(layer+1))
            D_BN = getattr(self, 'D_'+str(layer+1)+'_BN')
            x = D(x)
            x = D_BN(x)
            x = f.relu(x)
            residuals.append(x)
            outputs['D_'+str(layer+1)] = x

        for layer in reversed(range(self.nlayers)):
            G = getattr(self, 'G_'+str(layer+1))
            if layer < self.nlayers-1:
                x += residuals[layer]
            x = G(x)
            if layer > 0:
                G_BN = getattr(self, 'G_'+str(layer+1)+'_BN')
                x = G_BN(x)
                x = f.relu(x)
            outputs['G_'+str(layer+1)] = x

        result = (x, outputs) if all_layers else x
        return result

    
class CortexNetBase(nn.Module):
    '''
    A slightly modified version of cortexnet model02.
    No classifier attached. 
    '''
    KERNEL_SIZE = 3
    PADDING = KERNEL_SIZE // 2
    KERNEL_STRIDE = 2

    def __init__(self, network_size):
        super().__init__()
        self.nlayers = len(network_size) - 1

        
        for layer in range(self.nlayers):
            # Create D[layer] block
            multiplier = 1 if layer < 1 else 2
            D = nn.Conv2d(in_channels = network_size[layer] * multiplier,
                          out_channels = network_size[layer + 1],
                          kernel_size = CortexNetBase.KERNEL_SIZE,
                          stride = CortexNetBase.KERNEL_STRIDE,
                          padding = CortexNetBase.PADDING)
            D_BN = nn.BatchNorm2d(network_size[layer+1])

            # Create G[layer] block
            multiplier = 1 if layer == self.nlayers-1 else 2
            G = nn.ConvTranspose2d(in_channels = network_size[layer+1] * multiplier,
                                   out_channels = network_size[layer],
                                   kernel_size = CortexNetBase.KERNEL_SIZE,
                                   stride = CortexNetBase.KERNEL_STRIDE,
                                   padding = CortexNetBase.PADDING,
                                   output_padding=CortexNetBase.PADDING)
            G_BN = nn.BatchNorm2d(network_size[layer])

            setattr(self, 'D_'+str(layer+1), D)
            setattr(self, 'D_'+str(layer+1)+'_BN', D_BN)
            setattr(self, 'G_'+str(layer+1), G)
            if layer > 0 : setattr(self, 'G_'+str(layer+1)+'_BN', G_BN)

    def forward(self, x, state, all_layers = False):

        residuals = []
        state = state or [None] * (self.nlayers - 1)
        outputs = OrderedDict()

        for layer in range(self.nlayers):
            D = getattr(self, 'D_'+str(layer+1))
            D_BN = getattr(self, 'D_'+str(layer+1)+'_BN')
            if layer > 0:
                if state[layer - 1] is None:
                    s = V(x.data.clone().zero_())
                else:
                    s = state[layer - 1]
                x = torch.cat((x,s), 1)

            x = D(x)
            x = D_BN(x)
            x = f.relu(x)
            residuals.append(x)
            outputs['D_'+str(layer+1)] = x

        for layer in reversed(range(self.nlayers)):
            G = getattr(self, 'G_'+str(layer+1))
            if layer < self.nlayers-1:
                x = torch.cat((x, residuals[layer]), 1)
            x = G(x)
            if layer > 0:
                G_BN = getattr(self, 'G_'+str(layer+1)+'_BN')
                x = G_BN(x)
                x = f.relu(x)
                state[layer - 1] = x
            outputs['G_'+str(layer+1)] = x

        result = (x, state, outputs) if all_layers else (x, state)
        return result


class CortexNetSeg(CortexNetBase):
    '''
    Base cortex net modified for next frame + segmentation pred
    (assuming atleast one decoder and one generator)
    '''
    def __init__(self, network_size):
        super().__init__(network_size)

        G_SEG = nn.ConvTranspose2d(in_channels = self.G_1.in_channels,
                                   out_channels = 2,
                                   kernel_size = CortexNetBase.KERNEL_SIZE,
                                   stride = CortexNetBase.KERNEL_STRIDE,
                                   padding = CortexNetBase.PADDING,
                                   output_padding=CortexNetBase.PADDING)        
        setattr(self, 'G_SEG', G_SEG)

    def forward(self, x, state, all_layers = False):

        x, state, outputs = super().forward(x, state, True)

        # segmentation g block's input is either the second last G or
        # output of D if only one D,G blocks
        seg_in = torch.cat((outputs['D_1'], outputs['G_2']), 1) if self.nlayers > 1 else outputs['D_1']
        mask = self.G_SEG(seg_in)
        outputs['G_SEG'] = mask

        result = (x, mask, state, outputs) if all_layers else (x, mask, state)
        return result
