import torch
import gym

from src.wrappers.wrapper import Wrapper

class TorchWrapper(Wrapper):
    def __init__(self, env):
        super(TorchWrapper, self).__init__(env)

    def step(self, actions):
        assert len(actions) == self.num_envs
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().numpy()
        observation, reward, done, info = self.env.step(actions)
        
        return  torch.tensor(observation).float(), \
                torch.tensor(reward).float(), \
                torch.tensor(done).float(), info

    def reset(self):
        observation = self.env.reset()
        return torch.tensor(observation).float()