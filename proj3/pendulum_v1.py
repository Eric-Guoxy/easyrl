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


class QNet(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_noisy=False):
        super().__init__()

        # Store parameters
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_noisy = use_noisy

        LayerType = lambda in_dim, out_dim: NoisyLinear(in_dim, out_dim, sigma_zero=0.5) if use_noisy else nn.Linear

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

class PiNet(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Save the parameters
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Used layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Non-linear function
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.gelu(x)

        # Hidden layers
        for _ in range(self.num_layers - 1):
            x = self.hidden_layer(x)
            x = self.gelu(x)
        
        # Output layer
        x = self.output_layer(x) # The final logits

        return x

class DDPG:
    """This class uses the Pathwise Derivative Policy Gradient method as the backbone algorithm."""

    def __init__(self, config):
        self.config = config

        self.target_q_net = QNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions, # config.n_states for state, config.n_actions for action
            hidden_dim=20,
            output_dim=1, # Output the Q(s, a)
            use_noisy=True
        )
        self.target_q_net.eval() # The parameters will not be updated during training

        self.q_net = QNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict()) # Q_hat = Q

        self.target_actor_net = PiNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions,
        )
        self.target_actor_net.eval()

        self.actor_net = PiNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions,
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
        self.target_q_net_1 = QNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_1.eval()

        self.q_net_1 = QNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_1.load_state_dict(self.q_net_1.state_dict())

        self.target_q_net_2 = QNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_2.eval()

        self.q_net_2 = QNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1,
            use_noisy=True
        )
        self.target_q_net_2.load_state_dict(self.q_net_2.state_dict())

        self.target_actor_net = PiNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions
        )
        self.target_actor_net.eval()

        self.actor_net = PiNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions
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

    def load_model(self, q_path_1="td3_pendulum_q_1.pth", q_path_2="td3_pendulum_q_2.pth", actor_path="ddpg_pendulum_actor.pth"):
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
        self.actor_net = PiNet(
            num_layers=3,
            input_dim=self.config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions
        )

        # The actual policy net to be updated -> \pi_{\theta}
        self.policy_net = PiNet(
            num_layers=3,
            input_dim=self.config.n_states,
            hidden_dim=20,
            output_dim=config.n_actions
        )

        # The critic net (used for calculation of the advantage)
        self.critic_net = PiNet(
            num_layers=3,
            input_dim=config.n_states+config.n_actions,
            hidden_dim=20,
            output_dim=1
        )

        # Memory buffer
        self.memory = deque(maxlen=config.memory_capacity)


class Config:
    def __init__(self, n_states, n_actions, q_lr, actor_lr, memory_capacity, gamma, max_episodes, max_steps, sample_batch_size, use_noisy, play, train, action_min, action_max):
        self.n_states = n_states
        self.q_lr = q_lr
        self.actor_lr = actor_lr
        self.n_actions = n_actions
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.sample_batch_size = sample_batch_size
        self.use_noisy = use_noisy
        self.play = play
        self.train = train
        self.action_min = action_min
        self.action_max = action_max


class PPO:
    def __init__(self, config):
        self.config = config


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