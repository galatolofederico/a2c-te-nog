import sys
import gym
import argparse
import json
import torch

from src.wrappers.wrapper import Wrapper
from src.wrappers.torch import TorchWrapper
from src.wrappers.vector import VectorWrapper
from src.wrappers.statswrapper import StatsWrapper
from gym.wrappers import FrameStack
from gym.wrappers import AtariPreprocessing

from src.polices.actorcritic import SharedActorCritic


from src.agents.a2c import A2C
from src.agents.a2c_nog import A2CNOG


from src.logger import Logger

from src.utils import evaluate, make_atari_subproc_vecenv

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)

    args = parser.parse_args()

    params = json.loads(open(args.checkpoint+"/args.json").read())
    
    env = gym.make(params["env_name"])
    if params["atari"]:
        env = AtariPreprocessing(env)
        env = FrameStack(env, params["framestack"])
        env = Wrapper(env)
    env = VectorWrapper(env)
    env = TorchWrapper(env)
    
    policy = getattr(sys.modules[__name__], params["policy_name"])(params["framestack"] if params["atari"] else env.state_size, env.action_size, continuous=params["continuous"], stochastic_value=params["sv"], feature_extraction=params["feature_extraction"])

    agent = getattr(sys.modules[__name__], params["agent_name"])(env, policy, **params)

    policy.load_state_dict(torch.load(args.checkpoint+"/model.pth"))


    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)[0]
            obs, rew, done, _ = env.step(action)
            env.render()