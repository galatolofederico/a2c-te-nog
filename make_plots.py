import os
import argparse
import matplotlib.pyplot as plt
import plotszoo

def plot_optim(args, type):
    tag = "%s-%s" % (type, args.tag)
    query = {"config.tag": {"$eq": tag}}
    data = plotszoo.data.WandbData(args.username, args.project, query, force_update=args.update, verbose=False)

    parameters = ["config/train_steps", "config/gamma", "config/lr", "config/max_clip_norm"]
    if   type == "A2C":
        parameters.extend(["config/ent_coef", "config/vf_coef"])
    elif type == "A2CTE":
        parameters.extend(["config/vf_coef", "config/target_entropy"])
    elif type == "A2CNOG":
        parameters.extend(["config/ent_coef"])
    elif type == "A2CTENOG":
        parameters.extend(["config/target_entropy"])
    else:
        raise Exception("Unknown type %s" % (type, ))

    data.pull_scalars()
    assert len(data.scalars) > 0, "No data, check the tag name"
    data.dropna(parameters+["summary/reward"])
    data.pull_series()
    data.create_scalar_from_series("start_time", lambda s: s["_timestamp"].min())
    
    fig, axes = plt.subplots(1, len(parameters), sharey=False)

    parallel_plot = plotszoo.scalars.ScalarsParallelCoordinates(data, parameters, "summary/reward")

    parallel_plot.plot(axes)

    fig.set_size_inches(20, 10)
    plt.savefig(os.path.join(args.output_directory, "%s_optim_parallel.png" % (type, )))


    fig, ax = plt.subplots()
    
    scatter_cumulative = plotszoo.scalars.ScalarsScatterCumulative(data, "start_time", "summary/reward")

    scatter_cumulative.plot(ax, sort=True)

    fig.set_size_inches(20, 10)
    plt.savefig(os.path.join(args.output_directory, "%s_optim_history.png" % (type, )))

    parameters.extend(["config/total_steps", "config/num_envs"])
    args_names = [p.split("/")[1].replace("_","-") for p in parameters]
    best_run = data.scalars["summary/reward"].idxmax()
    best_args = "".join(["--%s %s " % (n, data.scalars[k][best_run]) for n,k in zip(args_names, parameters)])
    best_args += "--agent %s --env %s" % (type, data.scalars["config/env_name"][best_run])
    print("#%s\n%s" % (type, best_args))

parser = argparse.ArgumentParser()

parser.add_argument("--output-directory", type=str, default="./plots")
parser.add_argument("--username", type=str, default="galatolo")
parser.add_argument("--project", type=str, default="a2c-te-nog")
parser.add_argument("--tag", type=str)
parser.add_argument("--update", action="store_true")

args = parser.parse_args()

if not os.path.isdir(args.output_directory): os.mkdir(args.output_directory)

plot_optim(args, "A2C")
plot_optim(args, "A2CTE")
plot_optim(args, "A2CNOG")
plot_optim(args, "A2CTENOG")

