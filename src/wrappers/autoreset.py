
from src.wrappers.wrapper import Wrapper

class AutoReset(Wrapper):
    def __init__(self, env):
        super(AutoReset, self).__init__(env)
    
    def step(self, *args, **kwargs):
        state, reward, done, info = self.env.step(*args, **kwargs)
        if done:
            state = self.env.reset()
        return state, reward, done, info

    def reset(self):
        return self.env.reset()