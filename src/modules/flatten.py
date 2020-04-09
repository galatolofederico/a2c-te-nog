import torch

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)