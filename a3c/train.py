import a3c

import ray
import torch

ray.init()
ray.worker._register_class(torch.Tensor, use_pickle=True)
agent = a3c.A3CAgent("Pong-v4", {"num_workers": 16, "model": {"grayscale": True,
         "zero_mean": False,
         "dim": 42}})
for i in range(2):
	agent.train()

# agent = a3c.A3CAgent("CartPole-v0", {"num_workers": 4})
# for i in range(2):
# 	agent.train()
