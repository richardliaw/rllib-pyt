from policy import Policy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM(Policy):

    def _init(self, inputs, num_outputs, options):
        fcnet_activation = options.get("fcnet_activation", "tanh")
        activation = None
        if fcnet_activation == "tanh":
            activation = nn.Tanh
        elif fcnet_activation == "relu":
            activation = nn.ReLU
        elif fcnet_activation == "elu":
        	activation = nn.ELU

        layers = []
        last_layer_size = inputs
        for i in range(4):
            layers.append(nn.Conv2d(last_layer_size, 32, 3, stride=2, padding=1))
            layers.append(activation())
            last_layer_size = 32


        self.hidden_layers = nn.Sequential(*layers)
        self.lstm = nn.LSTMCell(32 * 3, 256)

        self.logits = nn.Linear(256, num_outputs)
        self.value_branch = nn.Linear(256, 1)

    def forward(self, inputs):
    	x, (hx, cx) = inputs
    	res = self.hidden_layers(x)
    	res = res.view(-1, 32*3*3)
    	hx, cx = self.lstm(res, (hx, cx))
    	res = hx

    	return self.logits(res), self.value_branch(res), (hx, cx)


