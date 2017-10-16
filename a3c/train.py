import a3c

import ray
import torch

ray.init()
ray.worker._register_class(torch.Tensor, use_pickle=True)
agent = a3c.A3CAgent("CartPole-v0", {"num_workers": 4})
ray.worker
for i in range(50):
	agent.train()