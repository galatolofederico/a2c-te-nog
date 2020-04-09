import gym

class Wrapper:
    def __init__(self, env):
        self.env = env
    
    @property
    def state_size(self):
        if hasattr(self.env, "num_envs"):
            assert self.num_envs == self.observation_space.shape[0]
            return self.env.observation_space.shape[1]
        else:
            return self.env.observation_space.shape[0]
            
    @property
    def action_size(self):
        if isinstance(self.env.action_space, gym.spaces.tuple.Tuple):
            assert len(self.env.action_space) == self.num_envs
            if isinstance(self.env.action_space[0], gym.spaces.Discrete):
                return self.env.action_space[0].n
            else:
                return self.env.action_space[0].shape[0]
        else:
            return self.env.action_space.n
    
    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return getattr(self, attr)
        else:
            return getattr(self.env, attr)