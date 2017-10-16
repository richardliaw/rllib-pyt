from policy import Policy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Linear(Policy):

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [16, 16])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        activation = None
        if fcnet_activation == "tanh":
            activation = nn.Tanh
        elif fcnet_activation == "relu":
            activation = nn.ReLU

        layers = []
        last_layer_size = inputs
        for size in hiddens:
            layers.append(nn.Linear(last_layer_size, size))
            layers.append(activation())
            last_layer_size = size

        self.hidden_layers = nn.Sequential(*layers)

        self.logits = nn.Linear(last_layer_size, num_outputs)
        self.probs = nn.Softmax()
        self.value_branch = nn.Linear(last_layer_size, 1)
        self.setup_loss()


if __name__ == '__main__':
    net = Linear(10, 5)