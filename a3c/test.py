from linear import Linear
import gym
import numpy as np
from collections import defaultdict, namedtuple
from runner import discount
from envs import create_and_wrap

import ipdb

# env = gym.make("CartPole-v0")
env = create_and_wrap(lambda: gym.make("Pong-v4"), 
        {"grayscale": True,
         "zero_mean": False,
         "dim": 42})
Batch = namedtuple(
    "Batch", ["si", "a", "adv", "r", "v", "terminal", "features"])


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
    features = rollout["features"][0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, batch_v, rollout["terminal"],
                 features)


def rollout(pi, env):
    """Do a rollout.

    If random_stream is provided, the rollout will take noisy actions with
    noise drawn from that stream. Otherwise, no action noise will be added.
    """
    rollout = defaultdict(list)
    t = 0
    ob = (env.reset())
    features = pi.get_initial_features()
    for _ in range(2000):
        rets = pi.compute(ob, features)
        ac, value, features = rets[0], rets[1], rets[2:]
        rollout["obs"].append(ob)
        rollout["vs"].append(value)
        rollout["actions"].append(ac)
        ob, rew, done, _ = env.step(ac)
        rollout["rs"].append(rew)
        rollout["features"].append(features)
        t += 1
        if done:
            break
    print("Cur policy: ", len(rollout["obs"]))
    rollout["r"] = 0
    rollout["terminal"] = True
    return rollout

from clstm import LSTM

policy = LSTM(env.observation_space.shape, env.action_space.n)
print("Current Norm", sum(p.norm().data.numpy() for p in policy.parameters()))
import pickle
for i in range(5):
    data = rollout(policy, env)
    batch = process_rollout(data, 0.99)
    print(sum([p.norm() for p in policy.parameters()]))
    # model_state = pickle.dumps(policy.get_weights())
    # import ipdb; ipdb.set_trace()
    # grad, info = policy.compute_gradients(batch)
    # policy.apply_gradients(grad)
    # print(sum([p.norm() for p in policy.parameters()]))

    # model_state = pickle.loads(model_state)
    # policy.set_weights(model_state)
    # print(sum([p.norm() for p in policy.parameters()]))
    policy.model_update(batch)
    print(sum([p.norm() for p in policy.parameters()]))



