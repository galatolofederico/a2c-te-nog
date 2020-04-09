import gym

from src.wrappers.wrapper import Wrapper

class StatsWrapper(Wrapper):
    def __init__(self, env):
        super(StatsWrapper, self).__init__(env)
        self.accumulated_steps = 0

        self.current_stats = self.dict_init()
        self.stats = self.dict_init()

    def dict_init(self):
        return {
            "episode/reward": 0,
            "episode/length": 0,
            "steps/reward": 0,
            "steps/episodes": 0,
        }

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)
        assert len(reward) == self.num_envs
        
        self.current_stats["episode/reward"] += reward[0]
        self.current_stats["episode/length"] += 1
        
        self.current_stats["steps/reward"] += reward[0]
        self.current_stats["steps/episodes"] += 1 if done[0] else 0
        
        self.accumulated_steps += 1

        if done[0]:
            self.stats["episode/reward"] = self.current_stats["episode/reward"]
            self.stats["episode/length"] = self.current_stats["episode/length"]
            
            self.current_stats["episode/reward"] = 0
            self.current_stats["episode/length"] = 0
        
        if self.accumulated_steps >= 1000:
            self.stats["steps/reward"] = self.current_stats["steps/reward"]
            self.stats["steps/episodes"] = self.current_stats["steps/episodes"]

            self.current_stats["steps/reward"] = 0
            self.current_stats["steps/episodes"] = 0
            self.accumulated_steps = 0

        return observation, reward, done, info

    def reset(self):
        return self.env.reset()
