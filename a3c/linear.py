import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# Code adapted from ELF


# TODO(rliaw): RNN
# TODO(rliaw): logging
# TODO(rliaw): GPU
# TODO(parameters)
# TODO - maybe make it such that local calls do not force the tensor out of the wrapping, only during remote calls

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.volatile = False
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self._init(input_dim, output_dim, {})

    def set_volatile(self, volatile):
        ''' Set model to ``volatile``.

        Args:
            volatile(bool): indicating that the Variable should be used in inference mode, i.e. don't save the history.'''
        self.volatile = volatile

    def set_gpu(self, id):
        pass

    def var_to_np(self, var):
        # Assumes single input
        return var.data.numpy()[0]


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
        self.probs = nn.Softmax()
        self.value_branch = nn.Linear(last_layer_size, 1)
        self.setup_loss()

    def setup_loss(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def _policy_entropy_loss(self, ac_logprobs, advs):
        return -(advs * ac_logprobs).mean()

    def _backward(self, batch):
        # not sure if this takes tensors ...........

        # reinsert into graphs
        states, acs, advs, rs = self._convert_batch(batch)
        values, ac_logprobs, entropy = self._evaluate(states, acs)
        pi_err = self._policy_entropy_loss(ac_logprobs, advs)
        value_err = (values - rs).pow(2).mean()

        self.optimizer.zero_grad()
        overall_err = value_err + pi_err - entropy * 0.1
        overall_err.backward()

    def _evaluate(self, states, actions):
        res = self.hidden_layers(states)
        values = self.value_branch(res)
        logits = self.logits(res)
        log_probs = F.log_softmax(logits)
        probs = self.probs(logits)
        action_log_probs = log_probs.gather(1, actions.view(-1, 1))
        entropy = -(log_probs * probs).sum(-1).mean()
        return values, action_log_probs, entropy

    def forward(self, x):
        res = self.hidden_layers(x)
        logits = self.logits(res)
        value = self.value_branch(res)
        return logits, value

    ########### EXTERNAL API ##################

    def _convert_batch(self, batch):
        states = Variable(torch.from_numpy(batch.si).float())
        acs = Variable(torch.from_numpy(batch.a))
        advs = Variable(torch.from_numpy(batch.adv.copy()).float())
        advs = advs.view(-1, 1)
        rs = Variable(torch.from_numpy(batch.r.copy()).float())
        rs = rs.view(-1, 1)
        return states, acs, advs, rs

    def model_update(self, batch):
        """ Implements compute + apply """
        # TODO(rliaw): Pytorch has nice 
        # caching property that doesn't require 
        # full batch to be passed in - can exploit that
        self._backward(batch)
        self.optimizer.step()

    def compute_gradients(self, batch):
        self._backward(batch)
        # Note that return values are just references;
        # calling zero_grad will modify the values
        return [p.grad.data.numpy() for p in self.parameters()]

    def apply_gradients(self, grads):
        for g, p in zip(grads, self.parameters()):
            p.grad = Variable(torch.from_numpy(g))
        self.optimizer.step()

    def compute(self, observations, features):
        x = Variable(torch.from_numpy(observations).float())
        logits, values = self(x)
        samples = self.probs(logits.unsqueeze(0)).multinomial().squeeze()
        return self.var_to_np(samples), self.var_to_np(values)

    def compute_logits(self, observations):
        x = Variable(torch.from_numpy(observations).float())
        res = self.hidden_layers(x)
        return self.var_to_np(self.logits(res))

    def value(self, observations):
        x = Variable(torch.from_numpy(observations).float())
        res = self.hidden_layers(x)
        res = self.value_branch(res)
        return self.var_to_np(res)

    def get_weights(self):
        ## !! This only returns references to the data.
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_initial_features():
        return None

if __name__ == '__main__':
    net = Linear(10, 5)
