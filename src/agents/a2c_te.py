import torch

from src.agents.a2c import A2C
from src.utils import discount
from src.modules.distributionwrapper import DistributionWrapper

def scale_probs(p, e):
    ret = torch.zeros_like(p)
    max_ind = p.max(-1).indices
    ret = p + e.unsqueeze(1)/(p.shape[-1]-1)
    ret[torch.arange(0, p.shape[0]), max_ind] = p[torch.arange(0, p.shape[0]), max_ind] - e
    return ret

def atleast_entropy(probs, target_entropy):
    entropy = -(torch.log(probs)*probs).sum(dim=1)
    delta_entropy = entropy - target_entropy
    max_ind = probs.max(dim=-1).indices
    mask = torch.zeros_like(probs)
    mask[torch.arange(0, probs.shape[0]), max_ind] = 1

    coef = torch.log(probs[mask == 0]).mean(dim=-1) \
            - torch.log(probs[mask == 1])
    eps = delta_entropy/coef
    eps[entropy > target_entropy] = 0
    num_eps = torch.finfo(probs.dtype).eps

    probs = torch.clamp(scale_probs(probs, eps), min=num_eps, max=1-num_eps)
    
    return probs

class A2CTE(A2C):
    def __init__(self, *args, target_entropy=0.4, **kwargs):
        super(A2CTE, self).__init__(*args, **kwargs)
        self.target_entropy = target_entropy

    def select_action(self, state):
        policy = self.policy(state)

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
                values=policy["value"],
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

        loss = policy_loss + self.vf_coef*value_loss

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
                "env/advantage": advantages.mean().item(),
                "env/entropy": replay["entropy"].mean().item(),
                "env/entropy_s": replay["entropy_s"].mean().item()
            }
        }
    
