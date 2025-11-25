from leo_gym.rl_algorithms.h_ppo.losses import *

class CostP3OLoss(LossComponent):
    """
    P3O cost penalty loss for hybrid (discrete + continuous) actions.

    Uses the joint action importance ratio and a cost advantage:

        ratio_total = exp( logπ_new(a|s) - logπ_old(a|s) )
        surr_cadv   = E[ ratio_total * adv_cost ]
        Jc          = ep_cost - cost_limit

        L_cost      = kappa * ReLU( surr_cadv + Jc )

    You can then just add this loss on top of your normal PPO actor loss.
    """

    def __init__(self,
                    policy_net: T.nn.Module,
                    kappa: float,
                    cost_limit: float):
        """
        :param policy_net: policy network returning (dist_dis, dist_cont).
        :param kappa: penalty coefficient.
        :param cost_limit: allowed average cost threshold.
        """
        self.policy_net = policy_net
        self.kappa = kappa
        self.cost_limit = cost_limit

    def compute_loss(self, batch_data: dict, cost_batch_key_index:str) -> Tuple[T.Tensor, dict]:
        """
        :param batch_data: batch data (dict)
        :param cost_batch_key_index: Key index for cost term in batch data
        
        :returns: 
            - Cost loss (T.Tensor)
            - Plot Logging Info (Dict) 
        """
        states = batch_data['obs']
        actions_dis = batch_data['act_dis']
        actions_cont = batch_data['act_cont']
        old_logp_dis = batch_data['old_logp_dis']
        old_logp_cont = batch_data['old_logp_cont']
        adv_cost = batch_data["adv_cost_"+cost_batch_key_index]   # cost advantage A^C_t
        ep_cost = batch_data['ep_cost_'+cost_batch_key_index]    # scalar or tensor for J_c estimate

        # Forward pass through policy to get new log-probs
        dist_dis, dist_cont = self.policy_net(states)

        new_logp_dis = dist_dis.log_prob(actions_dis).squeeze()
        new_logp_cont = dist_cont.log_prob(actions_cont).sum(dim=-1)

        # Joint log-prob for hybrid action
        new_logp_total = new_logp_dis + new_logp_cont
        old_logp_total = old_logp_dis + old_logp_cont

        # Importance sampling ratio for joint action
        ratio_total = (new_logp_total - old_logp_total).exp()

        # Surrogate cost-advantage term
        surr_cadv = (ratio_total * adv_cost).mean()

        # Make ep_cost a scalar tensor on the correct device
        if isinstance(ep_cost, (float, int)):
            ep_cost_t = T.tensor(ep_cost,
                                    dtype=adv_cost.dtype,
                                    device=adv_cost.device)
        else:
            ep_cost_t = ep_cost.to(adv_cost.device).float()
            if ep_cost_t.ndim > 0:
                ep_cost_t = ep_cost_t.mean()

        # Constraint violation term Jc
        Jc = ep_cost_t - self.cost_limit

        # P3O cost penalty loss
        cost_loss = self.kappa * T.relu(surr_cadv + Jc)

        return cost_loss, {
            'p3o_cost_loss': cost_loss.item(),
            'p3o_surr_cadv': surr_cadv.item(),
            'p3o_Jc': Jc.item(),
            'p3o_kappa': self.kappa,
            'p3o_cost_limit': self.cost_limit,
        }



class CostCriticLoss(LossComponent):
    def __init__(self, value_net:T.nn.Module, policy_clip: float)-> None:
        super().__init__()
        self.value_net = value_net
        self.policy_clip = policy_clip

    def compute_loss(self, batch_data: dict, cost_batch_key_index:str) -> Tuple[T.Tensor,dict]:
        """
        :param batch_data: batch data (dict)
        :param cost_batch_key_index: Key name for cost term in batch data
        
        :returns: 
            - Cost loss (T.Tensor)
            - Plot Logging Info (Dict) 
        """

        states = batch_data['obs']
        returns = batch_data['returns']
        values_old = batch_data["values_cost_"+cost_batch_key_index]

        value_pred = T.squeeze(self.value_net(states))
        value_pred_clipped = values_old + (value_pred - values_old).clamp(-self.policy_clip, self.policy_clip)

        loss_unclipped = (value_pred - returns).pow(2)
        loss_clipped = (value_pred_clipped - returns).pow(2)
        critic_loss = T.max(loss_unclipped, loss_clipped).mean()
        
        return critic_loss, {'values_cost_'+cost_batch_key_index: critic_loss.item()}
