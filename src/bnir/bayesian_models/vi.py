import copy

import torch
import torch.nn as nn


class VariationalInferenceModel(nn.Module):
    def __init__(self, mu, sigma=None):
        super().__init__()
        self.mu = mu
        self.eps = 1e-6
        if sigma is None:
            self.sigma = copy.deepcopy(mu)
            with torch.no_grad():
                for p in self.sigma.parameters():
                    p.copy_(self.eps * torch.ones_like(p))
        else:
            self.sigma = sigma

    def forward(self, x, base_model):
        # sample theta ~ q(theta), i.e., theta = mu + sigma*z, z~N(0,I)
        with torch.no_grad():
            for p, p_mu, p_sigma in zip(
                base_model.parameters(),
                self.mu.parameters(),
                self.sigma.parameters(),
            ):
                z = torch.randn_like(p)
                p.copy_(p_mu + p_sigma.clamp(min=self.eps) * z)
        if not self.training:
            with torch.no_grad():
                out = base_model(**x, return_dict=True)
        else:
            out = base_model(**x, return_dict=True)
        return out
