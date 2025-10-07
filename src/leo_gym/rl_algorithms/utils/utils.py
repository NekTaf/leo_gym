# ===== Standard =====
# Standard library
import os
import types

# Third-party
# ===== Third-party =====
import torch as T
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch_geometric.nn.aggr.deep_sets import DeepSetsAggregation


class TanhTransform(D.transforms.Transform):
    domain = D.constraints.real
    codomain = D.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        eps = 1e-6
        y = T.clamp(y, -1 + eps, 1 - eps)
        return 0.5 * (T.log1p(y) - T.log1p(-y))

    def log_abs_det_jacobian(self, x, y):
        derivative = 1 - y.pow(2)
        derivative = T.clamp(derivative, min=1e-6)
        return -T.log(derivative)
    
    
class SquashedNormal(TransformedDistribution):
    """
    Just like TransformedDistribution(Normal, [TanhTransform()]),
    but with a working .mean and a custom .entropy().
    """
    def __init__(self, loc, scale, num_entropy_samples=1000):
        base = D.Normal(loc, scale)
        super().__init__(base, [TanhTransform()])
        self.num_entropy_samples = num_entropy_samples

    @property
    def mean(self):
        m = self.base_dist.mean
        for t in self.transforms:
            m = t._call(m)
        return m

    def entropy(self):
        # 1) entropy of the Gaussian
        base_ent = self.base_dist.entropy()

        # 2) approximate E[ log |det Jac| ] via sampling
        #    shape: (num_samples, *batch_shape)
        samples = self.base_dist.rsample((self.num_entropy_samples,))

        # apply transform
        y = T.tanh(samples)
        log_det = self.transforms[0].log_abs_det_jacobian(samples, y)

        # average over the samples dimension
        correction = log_det.mean(0)

        # subtract the change-of-variables term
        return base_ent - correction
    
    
def save_checkpoint(model,
                    directory_save:str,
                    )->None:
    checkpoint_file = os.path.join(directory_save, model.name + '.pth')
    T.save(model.state_dict(), checkpoint_file)
    
    return

def load_checkpoint(model, 
                    file_name:str, 
                    train:bool=False,
                    device:T.device=T.device('cpu'),
                    )->None:
    
    
    state_dict = T.load(file_name, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    
    if train:
        model.train()
    else:
        model.eval()

    return



def proc_env_states(states):
    
    
    return states 

