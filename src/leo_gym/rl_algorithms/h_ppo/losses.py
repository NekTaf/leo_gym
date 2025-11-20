# Standard library
from abc import ABC, abstractmethod
from typing import Tuple

# Third-party
import torch as T
import torch.nn as nn


class LossComponent(ABC):
    @abstractmethod
    def compute_loss(self, batch_data: dict
                     )-> Tuple[T.Tensor,dict]:
        """
        Return NN loss and dictionary 
        with losses for logging purposes 
        """
        pass


class ActorLoss(LossComponent):

    def __init__(self, 
                 policy_net, 
                 policy_clip: float, 
                 log_ent_coef: nn.Parameter, 
                 target_kl: float):
        
        self.policy_net = policy_net
        self.policy_clip = policy_clip
        self.log_ent_coef = log_ent_coef
        self.target_kl = target_kl

    def _process_action_losses(self, prob_ratio, advantage, entropy):
        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = (
            T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
            * advantage
        )

        log_ratio = (prob_ratio + 1e-8).log()
        approx_kl = ((prob_ratio - 1) - log_ratio).mean()

        ppo_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
        
        ent_coef = T.exp(self.log_ent_coef)
        entropy_bonus = -ent_coef * entropy
        
        total_loss = ppo_loss + entropy_bonus

        mask = (approx_kl <= self.target_kl).float().to(total_loss.device)
        actor_loss = mask * total_loss

        return (
            actor_loss,
            entropy_bonus.item(),
            approx_kl.item(),
        )

    def compute_loss(self, batch_data: dict):
        states = batch_data['obs']
        actions_dis = batch_data['act_dis']
        actions_cont = batch_data['act_cont']
        old_logp_dis = batch_data['old_logp_dis']
        old_logp_cont = batch_data['old_logp_cont']
        adv_rew = batch_data['adv_rew']

        dist_dis, dist_cont = self.policy_net(states)

        # Discrete branch
        new_logp_dis = dist_dis.log_prob(actions_dis).squeeze()
        ratio_dis = (new_logp_dis - old_logp_dis).exp()
        
        ent_dis = dist_dis.entropy().mean()
        actor_loss_dis, entropy_bonus_dis, approx_kl_dis = self._process_action_losses(
            prob_ratio=ratio_dis,
            advantage=adv_rew,
            entropy=ent_dis
        )

        # Continuous branch
        new_logp_cont = dist_cont.log_prob(actions_cont).sum(dim=-1)
        ratio_cont = (new_logp_cont - old_logp_cont).exp()
        ent_cont = dist_cont.entropy().mean()
        actor_loss_cont, entropy_bonus_cont, approx_kl_cont = self._process_action_losses(
            prob_ratio=ratio_cont,
            advantage=adv_rew,
            entropy=ent_cont
        )

        total_actor_loss = actor_loss_dis + actor_loss_cont

        return total_actor_loss, {
            'actor_loss_dis': actor_loss_dis.item(),
            'entropy_bonus_dis': entropy_bonus_dis,
            'approx_kl_dis': approx_kl_dis,
            'actor_loss_cont': actor_loss_cont.item(),
            'entropy_bonus_cont': entropy_bonus_cont,
            'approx_kl_cont': approx_kl_cont,
            'total_actor_loss':total_actor_loss.item()
        }

class CriticLoss(LossComponent):

    def __init__(self, value_net, policy_clip: float):
        self.value_net = value_net
        self.policy_clip = policy_clip

    def compute_loss(self, batch_data: dict) -> Tuple[T.Tensor,dict]:
        states = batch_data['obs']
        returns = batch_data['returns']
        values_old = batch_data['values_old']

        value_pred = T.squeeze(self.value_net(states))
        value_pred_clipped = values_old + (value_pred - values_old).clamp(-self.policy_clip, self.policy_clip)

        loss_unclipped = (value_pred - returns).pow(2)
        loss_clipped = (value_pred_clipped - returns).pow(2)
        critic_loss = T.max(loss_unclipped, loss_clipped).mean()
        
        return critic_loss, {'critic_loss': critic_loss.item()}

