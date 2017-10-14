import torch
import torch.nn as nn
from torch.autograd import Variable
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

    def to_var(self, x, *, requires_grad=False, dtype=None):
        use_dtype = dtype if dtype else self.dtype
        x = torch.from_numpy(x).type(use_dtype) #.unsqueeze(0)
        # import ipdb; ipdb.set_trace()
        return Variable(x, volatile=self.volatile, requires_grad=requires_grad)

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
        self.probs = nn.Softmax()
        self.value_branch = nn.Linear(last_layer_size, 1)
        self.setup_loss()

    def setup_loss(self):
        self.value_loss = nn.MSELoss().cuda()
        self.pi_loss = nn.NLLLoss().cuda()
        self.optimizer = torch.optim.Adam(self.parameters())

    def _policy_entropy_loss(self, logits, acs, adv):
        def bw_hook(grad_in):
            grad = grad_in.mul(adv)
            # clip??
            return grad
        acs = self.to_var(acs, dtype=torch.LongTensor)  # ..
        logits = self.to_var(logits, requires_grad=True)
        batch = acs.size(0)
        pi_err = self.pi_loss(logits, acs)
        ent_err = self.entropy(logits) / batch
        logits.register_hook(bw_hook)
        return pi_err, ent_err

    def _backward(self, batch):
        # not sure if this takes tensors ...........
        advs = self.to_var(batch["adv"].copy())
        pi_err, ent_err = self._policy_entropy_loss(batch["si"],
                                                   batch["a"],
                                                   advs)
        rs = self.to_var(batch["r"].copy())
        vs = self.to_var(batch["v"])
        value_err = self.value_loss(vs, rs)
        overall_err = value_err + pi_err
        overall_err += ent_err * 0.1
        overall_err.backward()

    def model_update(self, batch):
        """ Implements compute + apply """
        # TODO(rliaw): Pytorch has nice 
        # caching property that doesn't require 
        # full batch to be passed in....
        batch = batch._asdict()
        self.optimizer.zero_grad()
        self._backward(batch)
        self.optimizer.step()

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def compute_gradients(self, batch):
        self.optimizer.zero_grad()
        self._backward(batch)

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
        sample = self.probs(logits.unsqueeze(0)).multinomial().squeeze()
        info = {}
        info["logits"] = logits.data.numpy()
        # TODO(rliaw): find out good way to abstract this
        return sample.data.numpy()[0], info  # note that this is not multidimensional friendly...

    def value(self, obs):
        x = self.to_var(obs)
        res = self.hidden_layers(x)
        res = self.value_branch(res)
        return res.data.numpy()[0]

    def entropy(self, logits):
        return (logits * torch.exp(logits)).sum()

if __name__ == '__main__':
    net = Linear(10, 5)