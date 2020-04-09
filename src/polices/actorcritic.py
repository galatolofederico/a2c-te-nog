import torch
import sys

from src.modules.init_linear import InitLinear
from src.modules.init_conv2d import InitConv2d
from src.modules.flatten import Flatten


class MLPFeatureExtractor(torch.nn.Module):
    def __init__(self, state_size, n_hidden):
        super(MLPFeatureExtractor, self).__init__()
        self.state_size = state_size
        self.n_hidden = n_hidden

        self.net = torch.nn.Sequential(
            InitLinear(self.state_size, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


class NatureCNN(torch.nn.Module):
    def __init__(self, input_size, featrues_size):
        super(NatureCNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            InitConv2d(input_size, 32, 8, stride=4, padding=0, nonlinearity="relu", mode="orthogonal"),
            torch.nn.ReLU(),
            InitConv2d(32, 64, 4, stride=2, padding=0, nonlinearity="relu", mode="orthogonal"),
            torch.nn.ReLU(),
            InitConv2d(64, 32, 2, stride=1, padding=0, nonlinearity="relu", mode="orthogonal"),
            torch.nn.ReLU(),
        )

        self.ff = torch.nn.Sequential(
            InitLinear(32, featrues_size, nonlinearity="tanh"),
            torch.nn.Tanh(),
        )

    
    def forward(self, x):
        num_envs = x.shape[0]
        frames = x.shape[1]
        x = x.view(num_envs, frames, 32, 32)
        features = self.cnn(x).view(num_envs, 32)
        return self.ff(features)




class Actor(torch.nn.Module):
    def __init__(self, n_hidden, action_size, continuous):
        super(Actor, self).__init__()
        self.n_hidden = n_hidden
        self.action_size = action_size
        self.continuous = continuous

        self.net = torch.nn.Sequential(
            InitLinear(self.n_hidden, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
        )

        if self.continuous:
            self.mu_head = InitLinear(self.n_hidden, self.action_size, nonlinearity="linear")
            self.std_head = InitLinear(self.n_hidden, self.action_size, nonlinearity="linear")
        else:
            self.probs_head = torch.nn.Sequential(
                InitLinear(self.n_hidden, self.action_size, nonlinearity="linear"),
                torch.nn.Softmax(dim=1)
            )

    def forward(self, x):
        x = self.net(x)
        if self.continuous: return (self.mu_head(x), torch.abs(self.std_head(x)))
        else: return self.probs_head(x)


class Critic(torch.nn.Module):
    def __init__(self, n_hidden):
        super(Critic, self).__init__()
        self.n_hidden = n_hidden
        self.net = torch.nn.Sequential(
            InitLinear(self.n_hidden, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, 1, nonlinearity="linear")
        )

    def forward(self, x):
        return self.net(x)



class SharedActorCritic(torch.nn.Module):
    def __init__(self, state_size, action_size, **kwargs):
        super(SharedActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_hidden = 128
        self.continuous = kwargs["continuous"]
        
        self.features = getattr(sys.modules[__name__], kwargs["feature_extraction"])(self.state_size, self.n_hidden)

        self.actor = Actor(self.n_hidden, self.action_size, continuous=self.continuous)

        self.critic = Critic(self.n_hidden)


    def forward(self, x, variant="classical"):
        features = self.features(x)
        if variant == "classical":
            return dict(
                probs = self.actor(features),
                value = self.critic(features).squeeze(1)
            )
        elif variant == "nog":
            return dict(
                probs   = self.actor(features),
                probs_d = self.actor(features.detach()),
                value_d = self.critic(features.detach()).squeeze(1)
            )
        elif variant == "nog-sv":
            return dict(
                probs   = self.actor(features),
                value_d = self.critic(features.detach()).squeeze(1)
            )



class NonSharedActorCritic(torch.nn.Module):
    def __init__(self, state_size, action_size, **kwargs):
        super(NonSharedActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_hidden = 128

        self.actor = torch.nn.Sequential(
            InitLinear(self.state_size, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, self.action_size, nonlinearity="linear"),
            torch.nn.Softmax(dim=1)
        )

        self.critic = torch.nn.Sequential(
            InitLinear(self.state_size, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, self.n_hidden, nonlinearity="tanh"),
            torch.nn.Tanh(),
            InitLinear(self.n_hidden, 1, nonlinearity="linear")
        )

    def forward(self, x, variant="classical"):
        if variant == "classical":
            return dict(
                probs = self.actor(x),
                value = self.critic(x).squeeze(1)
            )
        else:
            raise Exception("Unsupported variant %s" % (variant))

