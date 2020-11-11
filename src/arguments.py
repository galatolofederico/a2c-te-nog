import argparse


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, required=True, dest="env_name")
    parser.add_argument('--atari', action="store_true")
    parser.add_argument('--do-framestack', action="store_true")
    
    parser.add_argument('--continuous', action="store_true")
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--agent', type=str, default="A2C", dest="agent_name")
    parser.add_argument('--policy', type=str, default="SharedActorCritic", dest="policy_name")
    parser.add_argument('--feature-extraction', type=str, default="MLPFeatureExtractor")
    parser.add_argument('--sv', action="store_true")
    parser.add_argument('--framestack', type=int, default=4)
    parser.add_argument('--log-steps', type=int, default=1000)

    parser.add_argument('--target-entropy', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--stats-alpha', type=float, default=1e-2)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--max-clip-norm', type=float, default=0.5)
    
    parser.add_argument('--callback-interval', type=float, default=1e3)
    parser.add_argument('--total-steps', type=float, default=1e6)
    parser.add_argument('--train-steps', type=float, default=4)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--prune-reward', type=float, default=float("nan"))

    parser.add_argument('--log', type=str, default="stdout")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--wandb-name', type=str, default="a2c-te-nog")


    parser.add_argument('--render', action="store_true")
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--skip-eval', action="store_true")
    

    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)

    return parser
    

def infer_args(args):
    args.total_steps = int(args.total_steps)
    args.train_steps = int(args.train_steps)
    args.callback_interval = int(args.callback_interval)

    if args.env_name == "pong":
        args.env_name = "PongNoFrameskip-v4"
        args.atari = True
    if args.env_name == "breakout":
        args.env_name = "BreakoutNoFrameskip-v4"
        args.atari = True
    if args.env_name == "lunarlander":
        args.env_name = "LunarLander-v2"
    if args.env_name == "cartpole":
        args.env_name = "CartPole-v0"
    if args.env_name == "carracing":
        args.env_name = "DiscreteCarRacing-v0"
        args.feature_extraction = "NatureCNN"
        args.do_framestack = True
    
    if args.atari and args.policy_name == "SharedActorCritic":
        args.policy_name = "SharedAtariActorCritic"
    
    if args.name == "":
        import time
        args.name = "%s_%s_%s_%s" % (time.time(), args.env_name, args.policy_name, args.agent_name)


    if args.agent_name == "A2CSV" or args.agent_name == "A2CNOGSV":
        args.sv = True
    
    #if args.agent_name == "A2CTE" or args.agent_name == "A2CNOGME":
    #    args.me = True
    

    return args