import torch

from src.wrappers.wrapper import Wrapper
from src.wrappers.torch import TorchWrapper

class TorchStepRunnerWrapper(Wrapper):
    def __init__(self, env, steps, continuous=False):
        assert isinstance(env, TorchWrapper)
        super(TorchStepRunnerWrapper, self).__init__(env)
        self.steps = steps
        self.continuous = continuous
        self.need_reset = True

    def run(self, action_fn):
        if self.need_reset: self.reset()
        self.clear()
        for i in range(self.steps):
            fn_out = action_fn(self.current_state)
            log_dict = None
            if isinstance(fn_out, tuple):
                action, log_dict = fn_out
            else:
                action = fn_out
            
            self.states[i] = self.current_state
            self.actions[i] = action

            observation, reward, done, info = self.env.step(action)
            self.current_state = observation
            
            self.next_states[i] = observation
            self.rewards[i] = reward
            self.dones[i] = done

            if log_dict is not None:
                for field, value in log_dict.items():
                    if field not in self.log_dict:
                        self.log_dict[field] = torch.zeros(self.steps, *value.shape)
                    self.log_dict[field][i] = value

        return dict(
            states  = self.states,
            actions = self.actions,
            next_states = self.next_states,
            rewards = self.rewards,
            dones = self.dones,
            **self.log_dict
        )

    def reset(self):
        self.need_reset = False
        self.current_state = self.env.reset()
        self.state_shape = self.current_state.shape[1:]

    def clear(self):
        self.states = torch.zeros(self.steps, self.num_envs, *self.state_shape)
        self.next_states = torch.zeros(self.steps, self.num_envs, *self.state_shape)
        
        if self.continuous:
            self.actions = torch.zeros(self.steps, self.num_envs, self.env.action_size)
        else:
            self.actions = torch.zeros(self.steps, self.num_envs)
        self.rewards = torch.zeros(self.steps, self.num_envs)
        self.dones = torch.zeros(self.steps, self.num_envs)

        self.log_dict = dict()