import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import numpy as np
from typing import Tuple


# --- Actor-Critic Network ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.6):
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Pendulum action is [-max_torque, max_torque] -> Tanh scales to [-1, 1]
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Action variance (log standard deviation)
        self.action_log_std = nn.Parameter(
            torch.ones(1, action_dim) * np.log(action_std_init)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, nn.init.calculate_gain("tanh"))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self):
        raise NotImplementedError("Call act or evaluate methods explicitly.")

    def act(self, state):
        """Get action and log prob for interaction"""
        action_mean = self.actor(state)
        action_std = self.action_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(
            axis=-1
        )  # Sum across action dim if > 1
        value = self.critic(state)

        return action.detach(), action_logprob.detach(), value.detach()

    def evaluate(self, state, action):
        """Evaluate state and action for update"""
        action_mean = self.actor(state)
        action_std = self.action_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        action_logprob = dist.log_prob(action).sum(axis=-1)
        dist_entropy = dist.entropy().sum(axis=-1)
        value = self.critic(state)

        return action_logprob, value, dist_entropy


# --- PPO Agent ---
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        gae_lambda,
        K_epochs,
        eps_clip,
        action_std_init=0.6,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(
            self.device
        )
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
                {
                    "params": self.policy.action_log_std,
                    "lr": lr_actor,
                },  # Learn std dev too
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(
            self.device
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state) -> Tuple[np.ndarray, float, float]:
        """Select action using old policy for interaction"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, value = self.policy_old.act(state)

        # Return numpy arrays for env interaction, keep values as scalar tensors
        return action.cpu().numpy(), action_logprob.cpu().item(), value.cpu().item()

    def update(self, memory):
        """Update policy using collected trajectory"""
        # Convert lists from memory to tensors
        old_states = torch.FloatTensor(np.array(memory["states"])).to(self.device)
        old_actions = torch.FloatTensor(np.array(memory["actions"])).to(self.device)
        old_logprobs = torch.FloatTensor(memory["logprobs"]).to(self.device)
        rewards = torch.FloatTensor(memory["rewards"]).to(self.device)
        dones = torch.FloatTensor(memory["dones"]).to(self.device)
        old_values = torch.FloatTensor(memory["values"]).to(self.device)

        # Calculate advantages and returns
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)

        # Calculate TD Residuals (Advantages) and Returns
        # We need to handle the terminal state properly
        next_value = 0
        next_non_terminal = 0  # 0 if next state is terminal

        # Calculate advantages and returns in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = old_values[t] if not dones[t] else 0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = old_values[t + 1] if not dones[t] else 0
                next_non_terminal = 1.0 - dones[t]

            delta = (
                rewards[t] + self.gamma * next_value * next_non_terminal - old_values[t]
            )
            advantages[t] = delta + self.gamma * self.gae_lambda * next_non_terminal * (
                advantages[t + 1] if t < len(rewards) - 1 else 0
            )

        # Calculate returns
        returns = advantages + old_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Match state_values tensor dimensions
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # Calculate value loss using returns
            value_loss = 0.5 * self.MseLoss(state_values, returns)

            # Final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2).mean()
                + value_loss
                - 0.01 * dist_entropy.mean()
            )

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, filepath):
        torch.save(self.policy_old.state_dict(), filepath)

    def load(self, filepath):
        self.policy.load_state_dict(
            torch.load(filepath, map_location=lambda storage, loc: storage)
        )
        self.policy_old.load_state_dict(
            torch.load(filepath, map_location=lambda storage, loc: storage)
        )
