# Solving the scalarization issues of Advantage-based Reinforcement Learning Algorithms

Repository for the paper [Solving the scalarization issues of Advantage-based Reinforcement Learning Algorithms](https://arxiv.org/abs/2004.04120)


## Installation 

Clone the repository

```
git clone https://github.com/galatolofederico/a2c-te-nog.git
cd a2c-te-nog
```

Create a `virtualenv` and install the requirements

```
virtualenv --python=python3.6 env
. ./env/bin/activate
pip install -r requirements.txt
```

Additional packages are needed to use `wandb` or `tensorboard`


## Usage

To run an experiment run

```
python run.py --env <environment_name> ....
```

Check [arguments.py](https://github.com/galatolofederico/a2c-te-nog/blob/master/src/arguments.pyd) to see which hyperparameters can be set as arguments

To create an Optuna study

```
python create_study.py --study-name <study_name>
```

The environment variable `OPTUNA_STORAGE` must be set to a valid Optuna storage

To run a trial from an Optuna study

```
python run_trial.py --study-name <study_name>
```

## Examples

To create an hyperparameters optimization for the agent `A2CTENOG` as in the paper

```
python create_study.py --env lunarlander --log stdout --agent A2CTENOG --study-name A2CTENOG-1 --prune-reward -500 --total-steps 3e4 --num-envs 8
```

To run a run from this study

```
python run_trial.py --study-name A2CTENOG-1
```

To run a specify experiment (for example the best one found for A2CTENOG in the paper)

```
python run.py --env lunarlander --log stdout --prune-reward -500 --total-steps 3e4 --num-envs 8 --agent A2CTENOG --target-entropy 0.0917 --lr 0.0002292 --max-clip-norm 0.3462 --train-steps 64 --gamma 0.999
```

## Citing

If you want to cite use you can use this BibTeX

```
@article{galatolo_a2c
,	author	= {Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola}
,	title	= {Solving the scalarization issues of Advantage-based Reinforcement Learning Algorithms}
,	year	= {2020}
}
```


## Contributions and license

The code is released as Free Software under the [GNU/GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license. Coping, adapting e republishing it is not only consent but also encouraged. 

For any further question feel free to reach me at  [federico.galatolo@ing.unipi.it](mailto:federico.galatolo@ing.unipi.it) or on Telegram  [@galatolo](https://t.me/galatolo)