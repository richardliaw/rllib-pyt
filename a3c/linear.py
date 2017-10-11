import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim
        self._init()


class Linear(nn.Module):

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        activation = None
        if fcnet_activation == "tanh":
            activation = lambda: nn.tanh()
        elif fcnet_activation == "relu":
            activation = lambda: nn.relu()

        layers = []
        previous = inputs
        for size in hiddens:
            self.hiddens.append(nn.Linear(previous, size))
            self.hiddens.append(activation)
            previous = size

        self.hidden_layers = nn.Sequential(*layers)

        self.policy_branch = nn.Linear(previous, num_outputs)
        self.value_branch = nn.Linear(previous, 1)