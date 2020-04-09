import numpy as np
import gym

from src.wrappers.wrapper import Wrapper

class VectorWrapper(Wrapper):
    def __init__(self, env):
        self.num_envs = 1
        low = np.repeat(env.observation_space.low[np.newaxis, ...], 1, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], 1, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

        super(VectorWrapper, self).__init__(env)

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions[0])
        
        return  np.atleast_2d(observation), \
                np.atleast_2d(reward), \
                np.atleast_2d(done), info

    def reset(self):
        observation = self.env.reset()
        return np.atleast_2d(observation)