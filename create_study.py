import optuna
import argparse
import os

from src.arguments import get_parser, infer_args


parser = get_parser()

parser.add_argument('--study-name', type=str, required=True)

parser.add_argument('--sampler', type=str, default="TPESampler")
parser.add_argument('--pruner', type=str, default="SuccessiveHalvingPruner")

args = parser.parse_args()
args = infer_args(args)


study = optuna.create_study(
    study_name=args.study_name,
    direction="maximize",
    sampler=getattr(optuna.samplers, args.sampler)(),
    pruner=getattr(optuna.pruners, args.pruner)(),
    storage=os.environ["OPTUNA_STORAGE"]
)

study.set_user_attr("args", vars(args))
