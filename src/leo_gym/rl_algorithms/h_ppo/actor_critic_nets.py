# Third-party
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.nn.aggr.deep_sets import DeepSetsAggregation

# Local
from leo_gym.rl_algorithms.utils.utils import (
    SquashedNormal,
    load_checkpoint,
    save_checkpoint,
)


class ObservationEncoder(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 net_arch=[64, 64, nn.Tanh],  # last element can be activation class
                 device: str = "cpu"):
        """
        net_arch:
            - e.g. (64, 64, nn.Tanh) or (64, 64, nn.ReLU)
            - or just (64, 64) -> defaults to Tanh
            - or a single int 64 -> [64] with Tanh
        """
        super().__init__()
        self.device = T.device(device)
        
        activation_cls = net_arch[-1]          
        hidden_sizes = net_arch[:-1]       

        layers = []
        in_dim = obs_dim
        gain = nn.init.calculate_gain('tanh')  # assuming Tanh-like scale

        for hidden_size in hidden_sizes:
            linear = nn.Linear(in_dim, hidden_size)
            nn.init.orthogonal_(linear.weight, gain=gain)
            linear.bias.data.zero_()
            layers.append(linear)
            layers.append(activation_cls())
            in_dim = hidden_size

        self.net = nn.Sequential(*layers).to(self.device)
        self.output_dim = in_dim  # last hidden layer size

    def forward(self, obs):
        if not isinstance(obs, T.Tensor):
            obs = T.as_tensor(obs, dtype=T.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        return self.net(obs)


# Policy 
class PolicyNetwork(nn.Module):
    save_checkpoint = save_checkpoint
    load_checkpoint = load_checkpoint

    def __init__(self,
                 state_dim,
                 lr,
                 action_type,          # assume always hybrid: [("continuous", cdim), ("discrete", ddim)]
                 device,
                 continuous_dist_cls,  # e.g. SquashedNormal, ctor taking (mu, std)
                 std_0,
                 observation_encoder,
                 net_arch=[64, 64, nn.Tanh],
                 name='policynet'):
        super().__init__()
        self.device = device
        self.action_type = action_type
        self.name = name
        self.continuous_dist_cls = continuous_dist_cls

        # Shared encoder
        self.obs_enc = observation_encoder(state_dim, net_arch, self.device)
        feature_dim = self.obs_enc.output_dim

        # figure out continuous action size from action_type
        cont_act_size = None
        disc_act_size = None
        for act_type, act_size in self.action_type:
            if act_type == "continuous":
                cont_act_size = act_size
            elif act_type == "discrete":
                disc_act_size = act_size

        assert cont_act_size is not None, "action_type must contain a 'continuous' entry."
        assert disc_act_size is not None, "action_type must contain a 'discrete' entry."

        # continuous + discrete heads (hybrid)
        self.mu = None
        self.log_std = None
        self.actions = None

        self.std_0 = T.as_tensor(std_0, dtype=T.float32)
        action_size = cont_act_size

        if self.std_0.dim() == 0:
            self.std_0 = T.full((action_size,), self.std_0.item(), dtype=T.float32)
        assert self.std_0.shape[0] == action_size, \
            f"Length of std_0 ({self.std_0.shape[0]}) must match continuous action size ({action_size})."
        log_std_0 = self.std_0.log()

        for act_type, act_size in self.action_type:
            if act_type == "continuous":
                self.mu = nn.Linear(feature_dim, act_size)
                nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
                self.mu.bias.data.zero_()

                # fixed, state-independent std for stability 
                self.log_std = nn.Parameter(log_std_0.clone())

            elif act_type == "discrete":
                self.actions = nn.Linear(feature_dim, act_size)
                nn.init.orthogonal_(self.actions.weight, gain=1.0)
                self.actions.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.to(self.device)

    def forward(self, state):
        x = self.obs_enc(state)

        dist_cont, dist_dis = None, None

        # hybrid: both branches always expected to exist
        for act_type, _ in self.action_type:
            if act_type == "continuous":
                mu = self.mu(x)
                log_std = T.clamp(self.log_std, -7., 1.)
                std = T.exp(log_std)
                dist_cont = self.continuous_dist_cls(mu, std)

            elif act_type == "discrete":
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
                 observation_encoder,
                 net_arch=[64, 64, nn.Tanh],
                 name='valuenet'):
        super().__init__()

        self.device = device
        self.name = name

        self.obs_enc = observation_encoder(state_dim, net_arch, self.device)
        feature_dim = self.obs_enc.output_dim

        self.v = nn.Linear(feature_dim, 1)

        gain = nn.init.calculate_gain('tanh')
        nn.init.orthogonal_(self.v.weight, gain=gain)
        self.v.bias.data.zero_()

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.to(self.device)

    def forward(self, state):
        x = self.obs_enc(state)
        return self.v(x)
