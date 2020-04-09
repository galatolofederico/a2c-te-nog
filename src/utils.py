import torch
import statistics
import gym

from src.wrappers.autoreset import AutoReset
from src.wrappers.subprocenv import SubProcEnv
from src.wrappers.wrapper import Wrapper
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack

def make_subproc_vecenv(env_name, num_envs):
    def make_env():
        env = gym.make(env_name)
        env = AutoReset(env)
        return env

    return SubProcEnv([make_env for _ in range(0, num_envs)])


def make_atari_subproc_vecenv(env_name, num_envs, framestack=4):
    def make_env():
        env = gym.make(env_name)
        env = AtariPreprocessing(env)
        env = FrameStack(env, framestack)
        env = AutoReset(env)
        return env

    return SubProcEnv([make_env for _ in range(0, num_envs)])

def discount(gamma, rewards, dones, bootstrap):
    R = bootstrap
    discounted = torch.zeros_like(rewards)
    length = discounted.size(0)
    for t in reversed(range(length)):
        R = R * (1.0 - dones[t])
        R = rewards[t] + gamma * R
        discounted[t] = R.squeeze(0) 

    return discounted


def evaluate(env, agent, trials=100):
    tot_rewards = []
    tot_lengths = []

    for i in range(0, trials):
        done = False
        obs = env.reset()
        length = 0
        reward = 0
        while not done:
            agent_out = agent.select_action(obs)
            if isinstance(agent_out, tuple):
                action = agent_out[0]
            else:
                action = agent_out
            obs, rew, done, _ = env.step(action)
            reward += rew.item()
            length += 1

        tot_lengths.append(length)
        tot_rewards.append(reward)
        if i % 10 == 0: print("Evaluation %d/%d" % (i, trials))
    
    return {
        "reward": statistics.mean(tot_rewards),
        "length": statistics.mean(tot_lengths)
    }