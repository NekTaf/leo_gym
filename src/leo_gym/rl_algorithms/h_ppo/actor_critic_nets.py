# Standard library
import os
import types

# Third-party
import torch as T
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch_geometric.nn.aggr.deep_sets import DeepSetsAggregation

# Local
from leo_gym.rl_algorithms.utils.utils import (
    SquashedNormal,
    load_checkpoint,
    save_checkpoint,
)


class PolicyNetwork(nn.Module):
    
    save_checkpoint = save_checkpoint
    load_checkpoint = load_checkpoint

    def __init__(self,
                 state_dim,
                 lr,
                 action_type,
                 device,
                 std_type,
                 use_squashed_gaussian,
                 cont_act_size,
                 std_0,
                 nn_size=256,
                 name='policynet'):
        
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.action_type = action_type
        self.name = name
        self.std_type = std_type
        self.use_squashed_gaussian = use_squashed_gaussian

        self.std_0 = T.as_tensor(std_0, dtype=T.float32)
        action_size = cont_act_size
        
        if self.std_0.dim() == 0:
            self.std_0 = T.full(
                (action_size,),
                self.std_0.item(),
                dtype=T.float32,
                device=self.std_0.device
            )

        assert self.std_0.shape[0] == action_size, \
            f"Length of std_0 ({self.std_0.shape[0]}) must match action size ({action_size})."

        log_std_0 = T.log(self.std_0)

        # === Feature extractors ===
        ds_features = state_dim[0][1][1]
        nds_features = state_dim[1][1][0]
        self.fcnds1 = nn.Sequential(
            nn.Linear(nds_features, nn_size),
            nn.Tanh()
        )
        self.ds1 = DeepSetsAggregation(
            local_nn=nn.Sequential(
                nn.Linear(ds_features, nn_size),
                nn.Tanh()
            ),
            global_nn=nn.Sequential(
                nn.Linear(nn_size, nn_size),
                nn.Tanh()
            )
        )
        self.common_layer = nn.Sequential(
            nn.Linear(2 * nn_size, nn_size),
            nn.Tanh()
        )

        gain = nn.init.calculate_gain('tanh')
        
        for layer in [self.fcnds1[0],
                      self.ds1.local_nn[0], 
                      self.ds1.global_nn[0],
                      self.common_layer[0]]:
            
            nn.init.orthogonal_(layer.weight, gain=gain)
            layer.bias.data.fill_(0.0)

        # ===== Policy heads =====
        self.mu = None
        self.log_std = None
        if isinstance(self.action_type, list):
            for act_type, act_size in self.action_type:
                if act_type == "continuous":
                    
                    # ===== Mean layer =====
                    self.mu = nn.Linear(nn_size, act_size)                        
                    nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
                    self.mu.bias.data.fill_(0.0)

                    # ===== Fixed Gaussian action noise =====
                    if self.std_type == 0:
                        self.log_std = nn.Parameter(log_std_0)
                                                                        
                    # ===== State-dependent Gaussian =====
                    elif self.std_type == 1:
                        self.log_std = nn.Linear(nn_size, act_size)
                        nn.init.orthogonal_(self.log_std.weight, gain=1.0)
                        self.log_std.bias.data = log_std_0

                elif act_type == "discrete":
                    self.actions = nn.Linear(nn_size, act_size)
                    nn.init.orthogonal_(self.actions.weight, gain=1.0)
                    self.actions.bias.data.fill_(0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.to(self.device)

    def forward(self, state):
        ds_state, nds_state = state[0], state[1]
        nds_latent = self.fcnds1(nds_state).squeeze()
        ds_latent = self.ds1(ds_state, dim=-2).squeeze()
        x = T.cat((nds_latent, ds_latent), dim=-1)
        x = self.common_layer(x)

        dist_cont, dist_dis = None, None
        if isinstance(self.action_type, list):
            for action_type, _ in self.action_type:
                if action_type == "continuous":
                    mu = self.mu(x)

                    if self.std_type == 0:
                        log_std = self.log_std
                    elif self.std_type == 1:
                        log_std = self.log_std(x)
                        
                    log_std = T.clamp(log_std, -7, 1)
                    std = T.exp(log_std)

                    if self.use_squashed_gaussian:
                        dist_cont = SquashedNormal(mu, std)

                    else:
                        dist_cont = Normal(mu, std)

                elif action_type == "discrete":
                    probs = F.softmax(self.actions(x), dim=-1)
                    dist_dis = Categorical(probs)

        return dist_dis, dist_cont
    
    
class ValueNetwork(nn.Module):
    
    save_checkpoint = save_checkpoint
    load_checkpoint = load_checkpoint

    def __init__(self,
                 state_dim,
                 lr,
                 device,
                 nn_size=256,
                 name='valuenet'):
        super(ValueNetwork, self).__init__()
        self.device = device
        self.name = name

        ds_features = state_dim[0][1][1]
        nds_features = state_dim[1][1][0]
        self.fcnds1 = nn.Sequential(nn.Linear(nds_features, nn_size), nn.Tanh())
        self.ds1 = DeepSetsAggregation(
            local_nn=nn.Sequential(nn.Linear(ds_features, nn_size), nn.Tanh()),
            global_nn=nn.Sequential(nn.Linear(nn_size, nn_size), nn.Tanh())
        )
        self.common_layer = nn.Sequential(nn.Linear(2*nn_size, nn_size), nn.Tanh())
        self.v = nn.Linear(nn_size, 1)

        gain = nn.init.calculate_gain('tanh')
        for layer in [self.fcnds1[0], self.ds1.local_nn[0], self.ds1.global_nn[0], self.common_layer[0], self.v]:
            nn.init.orthogonal_(layer.weight, gain=gain)
            layer.bias.data.fill_(0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.to(self.device)

    def forward(self, state):
        ds_state, nds_state = state[0], state[1]
        nds_latent = self.fcnds1(nds_state).squeeze()
        ds_latent = self.ds1(ds_state, dim=-2).squeeze()
        x = T.cat((nds_latent, ds_latent), dim=-1)
        x = self.common_layer(x)
        
        return self.v(x)

             
        