import numpy as np
from gym.envs.classic_control import MountainCarEnv

class EnergyMountainCarEnv(MountainCarEnv):
    def step(self, action):
        state, reward, done, info = super(EnergyMountainCarEnv, self).step(action)
        position, velocity = self.state
        height = np.sin(3*position) + 1
        
        m = 100
        potential_energy = m*self.gravity*height 
        kinetic_energy = 0.5*m*velocity**2
        energy = potential_energy + kinetic_energy
        
        reward = energy
        return state, reward, done, info



if __name__ == "__main__":
    import gym
    import src.environments
    import time
    env = gym.make("EnergyMountainCar-v0")
    state = env.reset()
    while True:
        env.render()
        state, reward, done, info = env.step(env.action_space.sample())
        print("state: %s reward: %s done: %s info: %s" % (state, reward, done, info))
        if done: state = env.reset()
        time.sleep(0.1)