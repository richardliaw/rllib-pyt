from linear import Linear
import gym
import numpy as np
from collections import defaultdict, namedtuple
from runner import discount


import ipdb

env = gym.make("CartPole-v0")
Batch = namedtuple(
    "Batch", ["si", "a", "adv", "r", "v", "terminal", "features"])


def rollout(pi, env):
    """Do a rollout.

    If random_stream is provided, the rollout will take noisy actions with
    noise drawn from that stream. Otherwise, no action noise will be added.
    """
    rollout = defaultdict(list)
    t = 0
    ob = (env.reset())
    for _ in range(500):
        ac, info = pi.compute_action(ob)
        v = pi.value(ob)
        rollout["obs"].append(ob)
        rollout["vs"].append(v)
        rollout["actions"].append(ac)
        rollout["logits"].append(info["logits"])
        ob, rew, done, _ = env.step(ac)
        rollout["rs"].append(rew)
        t += 1
        if done:
            break
    rollout["r"] = 0
    rollout["terminal"] = True
    return rollout


def process_rollout(rollout, gamma, lambda_=1.0):
    """Given a rollout, compute its returns and the advantage."""
    batch_si = np.asarray(rollout["obs"])
    batch_a = np.asarray(rollout["actions"])
    batch_v = np.asarray(rollout["vs"])
    rewards = np.asarray(rollout["rs"])
    vpred_t = np.asarray(rollout["vs"] + [rollout["r"]])

    rewards_plus_v = np.asarray(rollout["rs"] + [rollout["r"]])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # This formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)
    features = None
    return Batch(batch_si, batch_a, batch_adv, batch_r, batch_v, rollout["terminal"],
                 features)


policy = Linear(env.observation_space.shape[0], env.action_space.n)
print("Current Norm", sum(p.norm().data.numpy() for p in policy.parameters()))
data = rollout(policy, env)
batch = process_rollout(data, 0.99)
policy.model_update(batch)
print("After Norm", sum(p.norm().data.numpy() for p in policy.parameters()))
