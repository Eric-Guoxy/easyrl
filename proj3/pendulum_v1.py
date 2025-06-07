# This file implements the Pendulum-v0 RL project.
# This file will implement the following algorithms: 
# (1) DDPG 
# (2) TD3 
# (3) PPO
# (4) GRPO

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, sigma_zero=0.5):
        super(NoisyLinear, self).__init__()

        # Store parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma_zero = sigma_zero

        self.mu_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.sigma_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.register_buffer('epsilon_weight', torch.Tensor(output_dim, input_dim))

        self.mu_bias = nn.Parameter(torch.Tensor(output_dim))
        self.sigma_bias = nn.Parameter(torch.Tensor(output_dim))
        self.register_buffer('epsilon_bias', torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        mu_std = 1.0 / math.sqrt(self.input_dim)
        self.mu_weight.data.normal_(0, mu_std)
        self.mu_bias.data.normal_(0, mu_std)

        initial_sigma_value = self.sigma_zero / math.sqrt(self.input_dim)
        self.sigma_weight.data.fill_(initial_sigma_value)
        self.sigma_bias.data.fill_(initial_sigma_value)
        
        self.sample_noise()

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.mu_weight.device)
        return x.sign().mul(x.abs().sqrt())

    def sample_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.epsilon_weight.copy_(epsilon_out.ger(epsilon_in))
        self.epsilon_bias.copy_(epsilon_out)

    def forward(self, x):
        weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
        bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, weight, bias)


class Net(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_noisy=False):
        super().__init__()

        # Store parameters
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_noisy = use_noisy

        LayerType = lambda in_dim, out_dim: NoisyLinear(in_dim, out_dim, sigma_zero=0.5) if use_noisy else nn.Linear(in_dim, out_dim)

        # Used layers
        self.input_layer = LayerType(input_dim, hidden_dim)
        self.hidden_layer = LayerType(hidden_dim, hidden_dim)
        self.output_layer = LayerType(hidden_dim, output_dim)

        # Used non-linear function
        self.gelu = nn.GELU()

    def forward(self, x):
        # The input layer
        x = self.input_layer(x)
        x = self.gelu(x)

        # The hidden layers
        for _ in range(self.num_layers - 1):
            x = self.hidden_layer(x)
            x = self.gelu(x)
        
        # The final layer
        x = self.output_layer(x)

        return x
    
    def sample_noise(self):
        if self.use_noisy:
            self.input_layer.sample_noise()
            self.hidden_layer.sample_noise()
            self.output_layer.sample_noise()


class PPOPolicyNet(Net):
    """
    This net is used for the policy nets in PPO.
    Given the state tensor as input, output the mean and std
    of the output action distribution (assumed Gaussian).
    """

    def __init__(self, num_layers, input_dim, hidden_dim, action_dim, log_std_min=-20, log_std_max=2, **kwargs):
        super().__init__(num_layers=num_layers, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim, **kwargs)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Head for the mean of the action distribution
        self.mu_head = nn.Linear(hidden_dim, action_dim)

        # Head for the log standard deviation of the action distribution
        # Note: It's often better to learn log_std for stability and ensure std > 0
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Non-linear function
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.gelu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.gelu(layer(x))
        
        mu = self.mu_head(x)

        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std


class DDPG:
    """This class uses the Pathwise Derivative Policy Gradient method as the backbone algorithm."""

    def __init__(self, config):
        self.config = config

        self.target_q_net = Net(
            num_layers=3,
            input_dim=config.n_states+config.n_actions, # config.n_states for state, config.n_actions for action
            hidden_dim=20,
            output_dim=1, # Output the Q(s, a)
            use_noisy=True
        )
        self.target_q_net.eval() # The parameters will not be updated during training

        self.q_net = Net(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict()) # Q_hat = Q

        self.target_actor_net = Net(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions,
            use_noisy=False
        )
        self.target_actor_net.eval()

        self.actor_net = Net(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions,
            use_noisy=False
        )
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        # Optimizer
        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.q_lr)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=config.actor_lr)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Memory buffer
        self.memory = deque(maxlen=config.memory_capacity)

        # Non-linear function
        self.tanh = nn.Tanh() # Used for the final logits to restrain output within [-2, 2]

        self.train = True
    
    def select_action(self, state):
        """Use self.actor_net to output an action given a state."""
        with torch.no_grad():
            # Convert state to a Pytorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = self.actor_net(state_tensor)
            action = 2 * self.tanh(logits)
        
        # Exploration
        if self.train:
            noise = np.random.normal(0, self.config.action_noise_std, size=action.shape)
            action = np.clip(action + noise, self.config.action_min, self.config.action_max)

        return action.squeeze().cpu().item()
    
    def update(self):
        # Sample a batch from the memory
        if len(self.memory) < self.config.sample_batch_size:
            return
        
        transitions = random.sample(self.memory, self.config.sample_batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        if self.config.use_noisy and hasattr(self.q_net.input_layer, 'mu_weight'):
            device = self.q_net.input_layer.mu_weight.device
        elif hasattr(self.q_net.input_layer, 'weight'):
            device = self.q_net.input_layer.weight.device
        else: # Fallback
            device = next(self.q_net.parameters()).device

        # Convert to Pytorch tensors
        batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
        batch_action = torch.FloatTensor(list(batch_action)).unsqueeze(1).to(device) # Change shape [2] to [2, 1]
        batch_reward = torch.FloatTensor(list(batch_reward)).unsqueeze(1).to(device)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
        batch_done = torch.FloatTensor(list(batch_done)).unsqueeze(1).to(device)

        # Calculate current Q values
        q_input = torch.cat((batch_state, batch_action), dim=1)
        current_q_values = self.q_net(q_input)

        # Target Q values: y = r_i + Q_target(s_{i+1}, actor_target(s_{i+1}))
        with torch.no_grad():
            target_actions = 2 * self.tanh(self.target_actor_net(batch_next_state)) # pi_hat(s_{i+1})
            target_q_input = torch.cat((batch_next_state, target_actions), dim=1)
            target_q_values = batch_reward + (1 - batch_done) * self.config.gamma * self.target_q_net(target_q_input)
        
        # Compute loss
        q_loss = self.loss_fn(current_q_values, target_q_values)

        actor_actions = 2 * self.tanh(self.actor_net(batch_state))
        actor_q_input = torch.cat((batch_state, actor_actions), dim=1)
        actor_loss = -self.q_net(actor_q_input).mean()

        # Backpropagation
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # For soft updates, happened every time update() is called
        self._soft_target_q_net_update()
        self._soft_target_actor_net_update()

    def _soft_target_q_net_update(self):
        """Soft update target Q-net for more robustness."""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        
    def _soft_target_actor_net_update(self):
        """Soft update target actor-net for more robustness."""
        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
    
    def save_model(self, q_path="ddpg_pendulum_q.pth", actor_path="ddpg_pendulum_actor.pth"):
        """Saves all networks' state_dicts."""
        torch.save(self.q_net.state_dict(), q_path)
        torch.save(self.actor_net.state_dict(), actor_path)
        print(f"Q-net saved to {q_path}, actor-net saved to {actor_path}")

    def load_model(self, q_path="ddpg_pendulum_q.pth", actor_path="ddpg_pendulum_actor.pth"):
        """Loads all networks' state_dicts."""
        self.q_net.load_state_dict(torch.load(q_path))
        self.target_q_net.load_state_dict(self.q_net.state_dict()) 
        self.actor_net.load_state_dict(torch.load(actor_path))
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        print(f"Q-net loaded from {q_path}, actor-net loaded from {actor_path}")

    def set_eval_mode(self):
        """Sets all networks to evaluation mode."""
        self.q_net.eval()
        self.target_q_net.eval()
        self.actor_net.eval()
        self.target_actor_net.eval()
        self.train = False

    def set_train_mode(self):
        """Sets q_net and actor_net to train mode (target_nets stay in eval)."""
        self.q_net.train()   
        self.actor_net.train()
        self.train = True

class TD3:
    """
    The Twin Delayed DDPG algorithm.
    Add three new features above the naive DDPG algorithm:
        (1) Clipped double Q-learning
        (2) Delayed policy updates
        (3) Target policy smoothing
    """

    def __init__(self, config):
        self.config = config

        # TD3 uses two sets of (Q_target, Q_actual) to mitigate
        # the overestimation of Q
        self.target_q_net_1 = Net(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_1.eval()

        self.q_net_1 = Net(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())

        self.target_q_net_2 = Net(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_2.eval()

        self.q_net_2 = Net(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())

        self.target_actor_net = Net(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions,
            use_noisy=False
        )
        self.target_actor_net.eval()

        self.actor_net = Net(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions,
            use_noisy=False
        )
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        # Optimizer
        self.q_optimizer_1 = torch.optim.AdamW(self.q_net_1.parameters(), lr=config.q_lr)
        self.q_optimizer_2 = torch.optim.AdamW(self.q_net_2.parameters(), lr=config.q_lr)
        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=config.actor_lr)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Memory buffer
        self.memory = deque(maxlen=config.memory_capacity)

        # Non-linear function
        self.tanh = nn.Tanh()

        # Trained steps
        self.steps = 0

        self.train = True

    def select_action(self, state):
        with torch.no_grad():
            # Convert state to a Pytorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0) 
            logits = self.actor_net(state_tensor)
            action = 2 * self.tanh(logits)
        
        # Exploration
        if self.train:
            noise = np.random.normal(0, self.config.action_noise_std, size=action.shape)
            action = np.clip(action + noise, self.config.action_min, self.config.action_max)
        
        return action.squeeze().cpu().item()

    def _smoothed_target_action(self, state):
        """Implement the target policy smoothing feature."""
        target_action = 2 * self.tanh(self.target_actor_net(state))
        noise = (torch.randn_like(target_action) 
                 * self.config.policy_noise_std
                ).clamp(-self.config.noise_clip, self.config.noise_clip)
        target_action_noisy = (target_action + noise).clamp(
            self.config.action_min,
            self.config.action_max
        )
        return target_action_noisy
    
    def update(self):
        # Sample a single batches for two Q-nets to learn
        if len(self.memory) < self.config.sample_batch_size:
            return
        
        transitions = random.sample(self.memory, self.config.sample_batch_size)

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        if self.config.use_noisy and hasattr(self.q_net.input_layer, 'mu_weight'):
            device = self.q_net.input_layer.mu_weight.device
        elif hasattr(self.q_net.input_layer, 'weight'):
            device = self.q_net.input_layer.weight.device
        else: # Fallback
            device = next(self.q_net.parameters()).device

        # Convert to Pytorch tensors
        batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
        batch_action = torch.FloatTensor(list(batch_action)).unsqueeze(1).to(device) # Change shape [2] to [2, 1]
        batch_reward = torch.FloatTensor(list(batch_reward)).unsqueeze(1).to(device)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
        batch_done = torch.FloatTensor(list(batch_done)).unsqueeze(1).to(device)

        # Calculate current Q values
        q_input = torch.cat((batch_state, batch_action), dim=1)
        current_q_values_1 = self.q_net_1(q_input)
        current_q_values_2 = self.q_net_2(q_input)

        # Target Q values: y = r + gamma * (1 - d) * min_{i=1,2} Q_target_i(next_state, a_{TD3}(next_state))
        with torch.no_grad():
            target_actions = self._smoothed_target_action(batch_next_state)
            target_q_inputs = torch.cat((batch_next_state, target_actions), dim=1)
            target_q_values = batch_reward + self.config.gamma * (1.0 - batch_done) * torch.min(self.target_q_net_1(target_q_inputs), self.target_q_net_2(target_q_inputs))
        
        # Compute loss
        q_loss_1 = self.loss_fn(current_q_values_1, target_q_values)
        q_loss_2 = self.loss_fn(current_q_values_2, target_q_values)

        # Update Q-critics
        self.q_optimizer_1.zero_grad()
        q_loss_1.backward()
        self.q_optimizer_1.step()
        
        self.q_optimizer_2.zero_grad()
        q_loss_2.backward()
        self.q_optimizer_2.step()

        self.steps += 1

        if self.steps % self.config.policy_delay == 0:
            actor_actions = 2 * self.tanh(self.actor_net(batch_state))
            actor_q_input = torch.cat((batch_state, actor_actions), dim=1)
        
            # Standard: use q_net_1 for actor loss (using the min of the two Q-net will introduce extra variance)
            actor_loss = -self.q_net_1(actor_q_input).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_target_q_net_update()
            self._soft_target_actor_net_update()

    def _soft_target_q_net_update(self):
        """Soft update target Q-net for more robustness."""
        for target_param, param in zip(self.target_q_net_1.parameters(), self.q_net_1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

        for target_param, param in zip(self.target_q_net_2.parameters(), self.q_net_2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
        
    def _soft_target_actor_net_update(self):
        """Soft update target actor-net for more robustness."""
        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
    
    def save_model(self, q_path_1="td3_pendulum_q_1.pth", q_path_2="td3_pendulum_q_2.pth", actor_path="td3_pendulum_actor.pth"):
        """Saves all networks' state_dicts."""
        torch.save(self.q_net_1.state_dict(), q_path_1)
        torch.save(self.q_net_2.state_dict(), q_path_2)
        torch.save(self.actor_net.state_dict(), actor_path)
        print(f"Q-net-1 saved to {q_path_1}, Q-net-2 saved to {q_path_2}, actor-net saved to {actor_path}")

    def load_model(self, q_path_1="td3_pendulum_q_1.pth", q_path_2="td3_pendulum_q_2.pth", actor_path="td3_pendulum_actor.pth"):
        """Loads all networks' state_dicts."""
        self.q_net_1.load_state_dict(torch.load(q_path_1))
        self.q_net_2.load_state_dict(torch.load(q_path_2))
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict()) 
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict()) 
        self.actor_net.load_state_dict(torch.load(actor_path))
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        print(f"Q-net-1 loaded from {q_path_1}, Q-net-2 loaded from {q_path_2}, actor-net loaded from {actor_path}")

    def set_eval_mode(self):
        """Sets all networks to evaluation mode."""
        self.q_net_1.eval()
        self.q_net_2.eval()
        self.target_q_net_1.eval()
        self.target_q_net_2.eval()
        self.actor_net.eval()
        self.target_actor_net.eval()
        self.train = False

    def set_train_mode(self):
        """Sets q_net and actor_net to train mode (target_nets stay in eval)."""
        self.q_net_1.train()   
        self.q_net_2.train()   
        self.actor_net.train()
        self.train = True


class PPO:
    """The Proximal Policy Optimization algorithm."""

    def __init__(self, config):
        self.config = config

        # The actor net (used for 'demonstration') -> \pi_{\theta^\prime}
        self.actor_net = PPOPolicyNet(
            num_layers=3,
            input_dim=self.config.n_states,
            hidden_dim=64,
            action_dim=config.n_actions, # One for the mean, one for the std of the output distribution (Gaussian)
            use_noisy=False,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max
        )
        self.actor_net.eval()

        # The actual policy net to be updated -> \pi_{\theta}
        self.policy_net = PPOPolicyNet(
            num_layers=3,
            input_dim=self.config.n_states,
            hidden_dim=64,
            action_dim=config.n_actions,
            use_noisy=False,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max
        )

        # The critic net (used for estimation of V(s))
        self.critic_net = Net(
            num_layers=3,
            input_dim=config.n_states, # Input the state tensor
            hidden_dim=64,
            output_dim=1, # Output the estimation V_hat(s)
            use_noisy=False
        )

        # Device
        self.device = self.policy_net.input_layer.weight.device

        # Memory buffer
        self.rollout_batch = [] # Will hold tuples: (s, a, old_logp, v, r, done)

        # Optimizer
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.config.policy_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=self.config.critic_lr)

        # Non-linear function
        self.tanh = nn.Tanh()

        self.loss_fn = nn.MSELoss()

        self.train = True
    
    def _clean_rollout_batch(self):
        self.rollout_batch = []
    
    def select_action(self, state):
        # Convert state to a Pytorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu, log_std = self.actor_net(state_tensor)
        action_dist = torch.distributions.Normal(mu, torch.exp(log_std))

        raw_action = action_dist.sample()
        squashed = self.tanh(raw_action)
        action = squashed * self.config.action_max        

        logp_raw = action_dist.log_prob(raw_action)
        logp = (logp_raw - torch.log(1 - squashed.pow(2) + 1e-6)).sum(-1, keepdim=True)

        return action.cpu().squeeze().item(), logp.cpu().squeeze().item()
    
    def compute_advantages_gae(self, rewards_tensor, dones_tensor, values_tensor, last_bootstrap_value):
        """
        Computes Generalized Advantage Estimation (GAE) for a rollout.

        Args:
            rewards_tensor (torch.Tensor): Tensor of rewards for each step in the rollout. Shape: (num_steps, 1) or (num_steps,).
            dones_tensor (torch.Tensor): Tensor of done flags for each step. Shape: (num_steps, 1) or (num_steps,).
                                         done[t] is True if s_{t+1} is terminal.
            values_tensor (torch.Tensor): Tensor of V(s_t) for each state in the rollout. Shape: (num_steps, 1) or (num_steps,).
            last_bootstrap_value (torch.Tensor): Scalar tensor V(s_T) where s_T is the state after the last action.
                                                 This is 0 if the episode ended at s_T.

        Returns:
            advantages (torch.Tensor): Tensor of GAE advantages.
            returns (torch.Tensor): Tensor of GAE-based returns (advantages + values_tensor).
        """
        num_steps = rewards_tensor.shape[0]
        advantages = torch.zeros_like(rewards_tensor)
        gae_accumulator = 0.0

        gamma = self.config.gamma
        lambda_gae = self.config.lambda_gae

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value_t = last_bootstrap_value
            else:
                next_value_t = values_tensor[t + 1]
            
            # Mask for the terminal states
            mask_t = 1.0 - dones_tensor[t].float()

            # Calculate TD error (delta)
            delta_t = rewards_tensor[t] + gamma * next_value_t * mask_t - values_tensor[t]
            
            # Update GAE
            gae_accumulator = delta_t + gamma * lambda_gae * mask_t * gae_accumulator
            advantages[t] = gae_accumulator
        
        # The returns are targets for the value function update
        returns = advantages + values_tensor
        return advantages, returns
    
    def update(self, last_state, is_last_state_terminal):
        """Use the policy gradient algorithm to update."""
        device = self.device

        with torch.no_grad():
            last_value = 0.0 # Default for terminal state
            if not is_last_state_terminal:
                last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(device)
                last_value = self.critic_net(last_state_t).cpu().item()

        # Unpack rollout batch into tensors
        batch_state, batch_action, batch_old_logp, batch_value, batch_reward, batch_done = zip(*self.rollout_batch)

        # Convert to Pytorch tensors
        batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
        batch_action = torch.FloatTensor(np.array(batch_action)).to(device) # Change shape [2] to [2, 1]
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1) # Ensure the size is [N, action_dim]
        batch_old_logp = torch.FloatTensor(np.array(batch_old_logp)).unsqueeze(1).to(device)
        batch_value = torch.FloatTensor(np.array(batch_value)).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)

        # Compute advantages and returns
        advantages, returns = self.compute_advantages_gae(
            batch_reward, batch_done, batch_value, torch.tensor([[last_value]], device=device)
        )

        # Randomly shuffle the steps for all batches
        batch_size = batch_state.size(0)

        # For accumulating losses and entropy over epochs and mini-batches
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_mini_batches_processed = 0

        for _ in range(self.config.ppo_epochs): # Train for self.config.ppo_epochs epochs
            perm = torch.randperm(batch_size, device=device)

            batch_state = batch_state[perm]
            batch_action = batch_action[perm]
            batch_old_logp = batch_old_logp[perm]
            advantages = advantages[perm]
            returns = returns[perm]

            # Iterate over mini-batches of size M
            M = self.config.mini_batch_size
            for start in range(0, batch_size, M):
                end = start + M
                if end > batch_size:
                    end = batch_size

                s_mb = batch_state[start:end]
                a_mb = batch_action[start:end]
                old_logp_mb = batch_old_logp[start:end]
                adv_mb = advantages[start:end]
                ret_mb = returns[start:end]
            
                # The target function: 
                # J(\theta) = \sum_{(s_t, a_t)} min(ratio * advantage, clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)
                epsilon = self.config.epsilon

                # Compute SI_coef (Sampling Importance)
                policy_mu_mb, policy_log_std_mb = self.policy_net(s_mb)
                policy_action_dist_mb = torch.distributions.Normal(policy_mu_mb, torch.exp(policy_log_std_mb))

                # Calculate entropy for the current mini-batch
                entropy_mb = policy_action_dist_mb.entropy().mean()

                # Reconstruct the pre-tanh output of a_mb
                a_squashed_rec_mb = torch.clamp(a_mb / self.config.action_max, -1.0 + 1e-7, 1.0 - 1e-7)
                a_raw_rec_mb = torch.atanh(a_squashed_rec_mb)

                new_logp_raw_mb = policy_action_dist_mb.log_prob(a_raw_rec_mb)
                new_logp_mb = (new_logp_raw_mb - torch.log(1 - a_squashed_rec_mb.pow(2) + 1e-6)).sum(-1, keepdim=True)

                ratio = torch.exp(new_logp_mb - old_logp_mb)

                # Compute the loss 
                policy_loss = -torch.min(ratio * adv_mb, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv_mb).mean()
                
                value_pred = self.critic_net(s_mb)
                critic_loss = self.loss_fn(value_pred, ret_mb)

                # Optimization
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_mb.item()
                num_mini_batches_processed += 1
        
        # Actor net update
        self.actor_net.load_state_dict(self.policy_net.state_dict())
    
        # Clean up the rollout_batch
        self._clean_rollout_batch()

        avg_policy_loss = total_policy_loss / num_mini_batches_processed if num_mini_batches_processed > 0 else 0
        avg_critic_loss = total_critic_loss / num_mini_batches_processed if num_mini_batches_processed > 0 else 0
        avg_entropy = total_entropy / num_mini_batches_processed if num_mini_batches_processed > 0 else 0

        return avg_policy_loss, avg_critic_loss, avg_entropy

    def save_model(self, policy_path="ppo_pendulum_policy.pth", actor_path="ppo_pendulum_actor.pth", critic_path="ppo_pendulum_critic.pth"):
        """Saves all networks' state_dicts."""
        torch.save(self.policy_net.state_dict(), policy_path)
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)
        print(f"Policy-net saved to {policy_path}, actor-net saved to {actor_path}, critic-net saved to {critic_path}")

    def load_model(self, policy_path="ppo_pendulum_policy.pth", actor_path="ppo_pendulum_actor.pth", critic_path="ppo_pendulum_critic.pth"):
        """Loads all networks' state_dicts."""
        self.policy_net.load_state_dict(torch.load(policy_path))
        self.actor_net.load_state_dict(torch.load(actor_path))
        self.critic_net.load_state_dict(torch.load(critic_path))
        print(f"Policy-net loaded from {policy_path}, actor-net loaded from {actor_path}, critic-net loaded from {critic_path}")

    def set_eval_mode(self):
        """Sets all networks to evaluation mode."""
        self.policy_net.eval()
        self.actor_net.eval()
        self.critic_net.eval()
        self.train = False

    def set_train_mode(self):
        """Sets q_net and actor_net to train mode (target_nets stay in eval)."""
        self.policy_net.train()   
        self.critic_net.train()
        self.train = True


class Config:
    def __init__(self, **kwargs):
        # n_states
        # q_lr
        # actor_lr
        # n_actions
        # memory_capacity
        # gamma
        # max_episodes (name changed for PPO, GRPO)
        # max_steps (name changed for PPO, GRPO)
        # sample_batch_size (optinal for PPO, GRPO)
        # use_noisy
        # play
        # train
        # action_min
        # action_max
        for k, v in kwargs.items():
            setattr(self, k, v)


class DDPGConfig(Config):
    def __init__(self, tau, action_noise_std, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.action_noise_std = action_noise_std


class TD3Config(Config):
    def __init__(self, tau, action_noise_std, policy_noise_std, policy_delay, noise_clip, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.action_noise_std = action_noise_std
        self.noise_clip = noise_clip
        self.policy_noise_std = policy_noise_std
        self.policy_delay = policy_delay

class PPOConfig(Config):
    def __init__(self, mini_batch_size, total_steps, num_steps_per_rollout, epsilon, policy_lr, critic_lr, ppo_epochs, log_std_min=-20, log_std_max=2, lambda_gae=0.95, **kwargs):
        super().__init__(**kwargs)
        self.mini_batch_size = mini_batch_size
        self.total_steps = total_steps
        self.num_steps_per_rollout = num_steps_per_rollout
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.lambda_gae = lambda_gae
        self.policy_lr = policy_lr
        self.critic_lr = critic_lr
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs