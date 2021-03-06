import torch

from src.agents.a2c import A2C
from src.utils import discount
from src.modules.distributionwrapper import DistributionWrapper

class A2CNOGSV(A2C):
    def __init__(self, *args, **kwargs):
        super(A2CNOGSV, self).__init__(*args, **kwargs)
        assert self.policy.critic.stochastic

    def select_action(self, state):
        policy = self.policy(state, variant="nog-sv")
        mass = DistributionWrapper(torch.distributions.Categorical, policy["probs"])

        action = mass.sample()

        return action, \
            dict(
                log_probs=mass.log_prob(action),
                values=policy["value_d"],
                entropy=mass.entropy()
            )

    def learn_step(self, replay):
        policy = self.policy(replay["next_states"][-1])
        last_value = policy["value"]
        
        discounted_rewards = discount(self.gamma, replay["rewards"], 
                                     replay["dones"], last_value)
        discounted_rewards = discounted_rewards.detach()
        
        advantages = discounted_rewards - replay["values"]
        
        policy_loss = -(replay["log_probs"]*(advantages.detach())).mean()
        value_loss = (advantages).pow(2).mean()

        self.optimizer.zero_grad()
        policy_loss.backward()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_clip_norm)
        self.optimizer.step()
        
        self.num_updates += 1

        return {
            "scalars": {            
                "loss/policy": policy_loss.item(),
                "loss/value": value_loss.item(),
                "env/advantage": advantages.mean().item(),
                "env/entropy": replay["entropy"].mean().item(),
            }
        }