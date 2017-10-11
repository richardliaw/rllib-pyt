import torch
import torch.nn as nn
from torch.autograd import Variable
# Code adapted from ELF


# TODO(rliaw): RNN
# TODO(rliaw): logging
# TODO(rliaw): GPU
# TODO(parameters)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self._init(input_dim, output_dim, {})

    def set_volatile(self, volatile):
        ''' Set model to ``volatile``.

        Args:
            volatile(bool): indicating that the Variable should be used in inference mode, i.e. don't save the history.'''
        self.volatile = volatile

    def to_var(self, x):
        return Variable(x, volatile=self.volatile)

    def set_gpu(self, id):
        pass


class Linear(Model):

    def _init(self, inputs, num_outputs, options):
        hiddens = options.get("fcnet_hiddens", [256, 256])
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
        self.value_branch = nn.Linear(last_layer_size, 1)

    def setup_loss(self):
        V = batch[self.value_node]
        value_err = self.value_loss(V, Variable(batch["target"]))
        self._reg_backward(V)
        stats["predicted_" + self.value_node].feed(V.data[0])
        stats[self.value_node + "_err"].feed(value_err.data[0])
        self.loss = self.pi_loss

    def model_update(self, batch):
        """ Implements compute + apply """
        # TODO(rliaw): Pytorch has nice 
        # caching property that doesn't require 
        # full batch to be passed in....
        self.optimizer.zero_grad()
        self.loss.backward(batch)
        self.optimizer.step()

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def compute_gradients(self, batch):
        self.optimizer.zero_grad()
        self.loss.backward(batch)

        # Note that return values are just references;
        # calling zero_grad will modify the values
        return [p.grad for p in self.parameters]

    def apply_gradients(self, grads):
        for g, p in zip(grads, self.parameters):
            p.grad = g
        self.optimizer.step()

    def compute_logits(self, observations):
        x = self.to_var(observations)
        res = self.hidden_layers(x)
        return self.logits(res)

    def compute_action(self, observations):
        logits = self.compute_logits(observations)
        import ipdb; ipdb.set_trace()
        return torch.multinomial(logits, 1)

    def value(self, obs):
        x = self.to_var(obs)
        res = self.hidden_layers(x)
        res = self.value_branch(res)
        return res

    def entropy(self):
        pass

if __name__ == '__main__':
    net = Linear(10, 5)
    import ipdb; ipdb.set_trace()