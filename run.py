import sys
import gym

from src.wrappers.wrapper import Wrapper
from src.wrappers.torch import TorchWrapper
from src.wrappers.torchsteprunner import TorchStepRunnerWrapper
from src.wrappers.statswrapper import StatsWrapper
from gym.wrappers import AtariPreprocessing
from gym.wrappers import FrameStack

from src.polices.actorcritic import SharedActorCritic
from src.polices.actorcritic import NonSharedActorCritic


from src.agents.a2c import A2C
from src.agents.a2c_nog import A2CNOG
from src.agents.a2c_sv import A2CSV
from src.agents.a2c_nog_sv import A2CNOGSV
from src.agents.a2c_me import A2CME
from src.agents.a2c_me_nog import A2CMENOG


from src.logger import Logger

from src.arguments import get_parser, infer_args

from src.utils import evaluate, make_atari_subproc_vecenv


def run(args, callback=None):    
    if args.atari:
        env = gym.vector.make(args.env_name,
                          num_envs=args.num_envs,
                          wrappers=[
                              AtariPreprocessing,
                              lambda x: FrameStack(x, args.framestack, lz4_compress=False),
                          ]
        )
    elif args.do_framestack:
        env = gym.vector.make(args.env_name,
                          num_envs=args.num_envs,
                          wrappers=[
                              lambda x: FrameStack(x, args.framestack, lz4_compress=False),
                          ]
        )   
    else:
        env = gym.vector.make(args.env_name, num_envs=args.num_envs)
    
    env = StatsWrapper(env)
    env = TorchWrapper(env)
    env = TorchStepRunnerWrapper(env, args.train_steps, args.continuous)    

    logger = Logger(args.log.split(",") if len(args.log) > 0 else [], env, args)
    
    policy = getattr(sys.modules[__name__], args.policy_name)(
        args.framestack if args.atari or args.do_framestack else env.state_size, 
        env.action_size, continuous=args.continuous,
        stochastic_value=args.sv, feature_extraction=args.feature_extraction)
    agent = getattr(sys.modules[__name__], args.agent_name)(env, policy, **vars(args), logger=logger)

    agent.learn(args.total_steps, callback, args.callback_interval)

    results = None
    if not args.skip_eval:
        if args.atari:
            env = make_atari_subproc_vecenv(args.env_name, 1)
        elif args.do_framestack:
            env = gym.vector.make(args.env_name,
                            num_envs=1,
                            wrappers=[
                                lambda x: FrameStack(x, args.framestack, lz4_compress=False),
                            ]
            )  
        else:
            env = gym.vector.make(args.env_name, num_envs=1)
        env = TorchWrapper(env)
        results = evaluate(env, agent, args.eval_steps)
        logger.log_results(results)
    
    if args.save:
        logger.save_model(policy)
    
    return results

if __name__ == "__main__":
    argparser = get_parser()
    args = argparser.parse_args()
    args = infer_args(args)
    run(args)
