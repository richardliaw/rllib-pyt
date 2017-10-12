from linear import Linear
import gym
import numpy as np

env = gym.make("CartPole-v0")

def rollout(pi, env):
    """Do a rollout.

    If random_stream is provided, the rollout will take noisy actions with
    noise drawn from that stream. Otherwise, no action noise will be added.
    """
    rews = []
    t = 0
    obs = []
    ob = (env.reset())
    for _ in range(500):
        import ipdb; ipdb.set_trace()
        ac = pi.compute_action(ob)[0]
        obs.append(ob)
        ob, rew, done, _ = env.step(ac)
        rews.append(rew)
        t += 1
        if done:
            break
    rews = np.array(rews, dtype=np.float32)
    return rews, t, np.array(obs)

policy = Linear(env.observation_space.shape[0], env.action_space.n)
rollout(policy, env)
