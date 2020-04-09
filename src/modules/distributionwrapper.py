import torch


class DistributionWrapper:
    def __init__(self, dist, *args):
        self.dist = dist
        if self.dist == torch.distributions.Categorical:
            self.probs = args[0]
            self.mass = self.dist(self.probs)
        elif self.dist == torch.distributions.Normal:
            self.mu, self.std = args
            self.mass = self.dist(self.mu, self.std)
        else:
            raise Exception("Unknown distribution with args %s" % (args))

    def sample(self):
        if self.dist == torch.distributions.Categorical:
            return self.mass.sample()
        elif self.dist == torch.distributions.Normal:
            return self.mass.rsample()

    def entropy(self, probs=None):
        if probs == None:
            if self.dist == torch.distributions.Categorical:
                return self.mass.entropy()
            if self.dist == torch.distributions.Normal:
                return self.mass.entropy().sum(dim=-1)
        else:
            if self.dist == torch.distributions.Categorical:
                probs = probs
                return -(torch.log(probs) * probs).sum(dim=-1)


    def log_prob(self, actions, probs=None):
        if probs is None:
            if self.dist == torch.distributions.Categorical:
                return self.mass.log_prob(actions)
            elif self.dist == torch.distributions.Normal:
                return self.mass.log_prob(actions).sum(dim=-1)
        else:
            if self.dist == torch.distributions.Categorical:
                eps = torch.finfo(probs.dtype).eps
                probs = probs.clamp(min=eps, max=1 - eps)
                return torch.log(probs)[torch.arange(0, probs.shape[0]),actions]