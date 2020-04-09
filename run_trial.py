import optuna
import copy
import argparse
import os


from run import run


def sample_params(trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.99, 0.999])
    train_steps = trial.suggest_categorical("train_steps", [8, 16, 32, 64])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    max_clip_norm = trial.suggest_uniform("max_clip_norm", 0, 2)

    hp = {
        "train_steps": train_steps,
        "gamma": gamma,
        "lr": lr,
        "max_clip_norm": max_clip_norm        
    }

    if args.agent_name == "A2C":
        if args.policy_name == "SharedActorCritic":
            ent_coef = trial.suggest_loguniform("ent_coef", 0.0001, 0.1)
            vf_coef = trial.suggest_uniform("vf_coef", 0, 1)

            hp.update({
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
            })
        elif args.policy_name == "NonSharedActorCritic":
            ent_coef = trial.suggest_loguniform("ent_coef", 0.0001, 0.1)

            hp.update({
                "ent_coef": ent_coef,
            })
        else:
            raise Exception("Unknown policy name %s"%(args.policy_name))

    if args.agent_name == "A2CNOG":
        ent_coef = trial.suggest_loguniform("ent_coef", 0.0001, 0.1)

        hp.update({
            "ent_coef": ent_coef,
        })

    if args.agent_name == "A2CTE":
        vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
        target_entropy = trial.suggest_uniform("target_entropy", 0, 0.2)
        
        hp.update({
            "vf_coef": vf_coef,
            "target_entropy": target_entropy
        })

    if args.agent_name == "A2CTENOG":
        target_entropy = trial.suggest_uniform("target_entropy", 0, 0.2)
        
        hp.update({
            "target_entropy": target_entropy
        })
    
    return hp



def objective(trial):
    hyperparameters = sample_params(trial)
    trial_args = copy.deepcopy(args)

    vars(trial_args).update(hyperparameters)
    trial_args.name = trial_args.name+"_"+str(trial.number)
    
    def callback(step, stats):
        print("--- Callback ---")
        trial.report(stats["stats/reward_mean"], step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    results = run(trial_args, callback)
    return results["reward"]


parser = argparse.ArgumentParser()

parser.add_argument('--study-name', type=str, required=True)

args = parser.parse_args()

study = optuna.load_study(study_name=args.study_name, storage=os.environ["OPTUNA_STORAGE"])

vars(args).update(study.user_attrs["args"])
study.optimize(objective, n_trials=1)