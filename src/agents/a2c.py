import torch
import math

from src.wrappers.torchsteprunner import TorchStepRunnerWrapper
from src.utils import discount
from src.modules.distributionwrapper import DistributionWrapper

class A2C:
    def __init__(self, env, policy, gamma=0.99, vf_coef=0.5, 
                ent_coef=0.01, max_clip_norm=0.5, lr=7e-4,
                eps=1e-5, alpha=0.99, logger=None,
                prune_reward=float("nan"), stats_alpha=1e-2,
                continuous=False, **kwargs):
        super(A2C, self).__init__()
        
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_clip_norm = max_clip_norm
        self.continuous = continuous
        self.prune_reward = prune_reward
        self.stats_alpha = stats_alpha

        self.lr = lr
        self.eps = eps
        self.alpha = alpha

        self.logger = logger
        self.num_updates = 0
        
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=self.lr, eps=self.eps, alpha=self.alpha)

    def select_action(self, state):
        policy = self.policy(state)
        mass = DistributionWrapper(torch.distributions.Categorical, policy["probs"])
        action = mass.sample()

        return action, \
            dict(
                log_probs=mass.log_prob(action),
                values=policy["value"],
                entropy=mass.entropy()
            )


    def learn(self, steps, callback=None, callback_interval=None):
        assert isinstance(self.env, TorchStepRunnerWrapper)
        warmup = 1e2
        delta_step = 1e3
        reward_mean = None
        last_delta = None
        for step in range(steps):
            replay = self.env.run(self.select_action)
            stats = self.learn_step(replay)["scalars"]
            if hasattr(self.env, "stats"):
                stats.update(self.env.stats)
            if step >= warmup:
                if reward_mean is None:
                    reward_mean = stats["episode/reward"]
                    stats["stats/reward_mean"] = reward_mean
                else:
                    reward_mean = self.stats_alpha*stats["episode/reward"] + (1-self.stats_alpha)*reward_mean
                    stats["stats/reward_mean"] = reward_mean
            if step % delta_step == 0:
                if last_delta is None:
                    last_delta = reward_mean
                else:
                    stats["stats/delta"] = reward_mean - last_delta
            
            if self.logger is not None:
                self.logger.log_scalars(stats)

            if callback is not None:
                if step > 0 and step % callback_interval == 0:
                    callback(step, stats)
            for key, value in stats.items():
                if math.isnan(value):
                    return
            if not math.isnan(self.prune_reward) and reward_mean is not None:
                if reward_mean < self.prune_reward:
                    return

    def learn_step(self, replay):
        policy = self.policy(replay["next_states"][-1])
        last_value = policy["value"]
        discounted_rewards = discount(self.gamma, replay["rewards"], 
                                     replay["dones"], last_value)
        discounted_rewards = discounted_rewards.detach()
        
        advantages = discounted_rewards - replay["values"]
        
        entropy_loss = replay["entropy"].mean()
        policy_loss = -(replay["log_probs"]*(advantages.detach())).mean()
        value_loss = (advantages).pow(2).mean()

        loss = policy_loss + self.vf_coef*value_loss - self.ent_coef*entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_clip_norm)
        self.optimizer.step()
        
        self.num_updates += 1

        return {
            "scalars": {            
                "loss/loss": loss.item(),
                "loss/policy": policy_loss.item(),
                "loss/value": value_loss.item(),
                "loss/entropy": entropy_loss.item(),
                "env/advantage": advantages.mean().item(),
            }
        }
    
        