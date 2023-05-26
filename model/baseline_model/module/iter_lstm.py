# -*- coding: utf-8 -*-
import torch
from torch import nn


class IterativeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, c0=None, h0=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # if c0 is not None:
        #     self.c0 = c0
        # else:
        #     self.c0 = torch.randn(hidden_dim)
        # if h0 is not None:
        #     self.h0 = h0
        # else:
        #     self.h0 = torch.randn(hidden_dim)
        
        self.xi = nn.Linear(input_dim, hidden_dim)
        self.hi = nn.Linear(hidden_dim, hidden_dim)
        self.xf = nn.Linear(input_dim, hidden_dim)
        self.hf = nn.Linear(hidden_dim, hidden_dim)
        self.xo = nn.Linear(input_dim, hidden_dim)
        self.ho = nn.Linear(hidden_dim, hidden_dim)
        self.xg = nn.Linear(input_dim, hidden_dim)
        self.hg = nn.Linear(hidden_dim, hidden_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_current, c_previous, h_previous,):
        # print("lstm")
        input_gate = self.sigmoid(self.xi(x_current) + self.hi(h_previous))
        forget_gate = self.sigmoid(self.xf(x_current) + self.hf(h_previous))
        output_gate = self.sigmoid(self.xo(x_current) + self.ho(h_previous))
        g_current = self.tanh(self.xg(x_current) + self.hg(h_previous))

        forgetten_tensor = forget_gate * c_previous
        inputting_tensor = input_gate * g_current
        c_current = forgetten_tensor + inputting_tensor
        h_current = output_gate * self.tanh(c_current)

        return c_current, h_current
