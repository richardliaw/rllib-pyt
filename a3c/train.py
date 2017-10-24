import a3c

import ray
import torch

ray.init(redirect_output=False)
d = {"num_workers": 16,
     "num_batches_per_iteration": 100,
     "model": {"grayscale": True,
         "zero_mean": False,
         "dim": 42}}
agent = a3c.A3CAgent("Pong-v4", d )
for i in range(50):
	agent.train()

# agent = a3c.A3CAgent("CartPole-v0", {"num_workers": 4})
# for i in range(2):
# 	agent.train()
