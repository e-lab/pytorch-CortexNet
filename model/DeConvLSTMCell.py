import torch
from torch import nn
from torch.nn import functional as f
from torch.autograd import Variable as V

KERNEL_SIZE = 3
STRIDE = 2
PADDING = KERNEL_SIZE // 3

# Note to self: Cannot convinvince myself that these ops on a single convolution is valid
class DeConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x_gate = nn.ConvTranspose2d(in_channels = input_size,
                                         out_channels = hidden_size,
                                         kernel_size = KERNEL_SIZE,
                                         stride = STRIDE,
                                         padding = PADDING)
        self.h_gate = nn.ConvTranspose2d(in_channels = input_size,
                                         out_channels = hidden_size,
                                         kernel_size = KERNEL_SIZE,
                                         stride = STRIDE,
                                         padding = PADDING)
        
    def forward(self, x, state):
        
        if state is None:
            state_size = x.data.size()
            state_size[1] = self.hidden_size
            state_size[2] = state_size[2]*STRIDE - 2*PADDING + KERNEL_SIZE
            state_size[3] = state_size[3]*STRIDE - 2*PADDING + KERNEL_SIZE
            state = [V(torch.zeros(state_size)), V(torch.zeros(state_size))]
            
        h, c = state

        x = self.x_gate(x)
        h = self.h_gate(h)

        x_i, x_f, x_c, x_o = x.chunk(4, 1)
        h_i, h_f, h_c, h_o = h.chunk(4, 1)

        i_t = f.sigmoid(x_i + h_i)
        f_t = f.sigmoid(x_f + h_f)
        c_t = f_t * c + i_t * f.tanh(x_c + h_c)
        o_t = f.sigmoid(x_o + h_o)
        h_t = o_t *  f.tanh(c_t)

        return h_t, (h_t, c_t)
