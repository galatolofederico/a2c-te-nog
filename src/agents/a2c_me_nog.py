import torch

from src.agents.a2c import A2C
from src.utils import discount
from src.modules.distributionwrapper import DistributionWrapper

from src.agents.a2c_me import atleast_entropy

class A2CMENOG(A2C):
    def __init__(self, *args, target_entropy=0.4, **kwargs):
        super(A2CMENOG, self).__init__(*args, **kwargs)
        self.target_entropy = target_entropy

    def select_action(self, state):
        policy = self.policy(state, variant="nog")

        probs = policy["probs"].detach()
        entropy = -(torch.log(probs)*probs).sum(dim=1)
        
        if torch.any(entropy < self.target_entropy):
            probs_s = atleast_entropy(probs, self.target_entropy)
        else:
            probs_s = probs
        entropy_s = -(torch.log(probs_s)*probs_s).sum(dim=1)
        
        probs_s = probs
        mass = DistributionWrapper(torch.distributions.Categorical, probs_s)
        action = mass.sample()

        return action, \
            dict(
                log_probs=mass.log_prob(action, policy["probs"]),
                values=policy["value_d"],
                entropy=entropy,
                entropy_s=entropy_s,
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
                "env/entropy_s": replay["entropy_s"].mean().item()
            }
        }
    
