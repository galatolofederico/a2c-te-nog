import numpy as np
from gym.envs.classic_control import MountainCarEnv

class EnergyMountainCarEnv(MountainCarEnv):
    def __init__(self, *args, **kwargs):
        super(EnergyMountainCarEnv, self).__init__(*args, **kwargs)
        self.solve_prize = 0.25
        self.max_steps = 1e3

    def energy(self, state):
        position, velocity = state
        height = self._height(position)

        m = 100
        potential_energy = m*self.gravity*height 
        kinetic_energy = 0.5*m*velocity**2
        
        return potential_energy + kinetic_energy

    def reset(self):
        state = super(EnergyMountainCarEnv, self).reset()
        self.current_energy = self.energy(state)
        self.steps = 0
        return state

    def step(self, action):
        state, _, solved, info = super(EnergyMountainCarEnv, self).step(action)
        new_energy = self.energy(state)
        self.steps += 1

        reward = new_energy - self.current_energy
        reward += self.solve_prize if solved else 0

        self.current_energy = new_energy
        return state, reward, solved or self.steps >= self.max_steps, info


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