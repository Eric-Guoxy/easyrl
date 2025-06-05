# This file implements the CartPole-v0 classic rl example.
# This program will use DDQN as the backbone RL algorithm.
import math
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque 
import random
import matplotlib.pyplot as plt
import wandb

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


class DDQN:
    def __init__(self, config):
        # Set up the two nets
        self.target_net = QNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=4*config.n_states,
            output_dim=config.n_actions,
            use_noisy=config.use_noisy
        )
        self.target_net.eval()

        self.policy_net = QNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=4*config.n_states,
            output_dim=config.n_actions,
            use_noisy=config.use_noisy
        )
        self.target_net.load_state_dict(self.policy_net.state_dict()) # The policy net and the target net should be the same at the beginning

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

        # Used to save transitions
        self.memory = deque(maxlen=config.memory_capacity)
        self.sample_batch_size = config.sample_batch_size

        self.config = config
        self.epsilon = config.epsilon

        # Keep record of the update steps
        self.step = 0

        self.train = True
    
    def select_action(self, state):
        """Given a state, output an action."""
        if random.random() < self.epsilon and self.train and not self.config.use_noisy:
            return random.randrange(self.config.n_actions)
        else:
            with torch.no_grad():
                # Convert state to tensor, add batch dimension
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_ations = self.policy_net(state_tensor)
                action = torch.argmax(q_ations).item()
            return action

    def update(self):
        # Step 1: Sample a batch from the memory
        if len(self.memory) < self.sample_batch_size:
            return # If there are not enough sampels, pass update

        transitions = random.sample(self.memory, self.sample_batch_size)
        
        # Use a batched version
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        device = self.policy_net.input_layer.mu_weight.device

        # Convert to Pytorch tensors
        batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
        batch_action = torch.LongTensor(np.array(batch_action)).unsqueeze(1).to(device) # Change shape [2] to [2, 1]
        batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
        batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)

        # Current Q values
        current_q_values = self.policy_net(batch_state).gather(1, batch_action) # current_q_values: [batch_size, n_actions], gather alone the '1' dimension -> [batch_size, 1]

        # Target Q values
        # For DDQN: Q_target = r + gamma * Q_target_net(s', argmax_a' Q_policy_net(s', a'))
        with torch.no_grad():
            batch_best_actions = torch.argmax(self.policy_net(batch_next_state), dim=1, keepdim=True) # Select the best actions along dim 1 -- [batch_size, n_actions]
            batch_next_state_q_values = self.target_net(batch_next_state).gather(1, batch_best_actions)
            
            target_q_values = batch_reward + (1 - batch_done) * self.config.gamma * batch_next_state_q_values # If done, the target Q value is merely the reward

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Backpropagation
        # Optinal: Clip gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # Update the parameters of the policy_net

        # Optinal: Decay the epsilon

        self.step += 1

        if self.config.epsilon > self.config.epsilon_min and not self.config.use_noisy:
            self.epsilon *= self.config.epsilon_decay

    def save_model(self, path="ddqn_cartpole.pth"):
        """Saves the policy network's state_dict."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="ddqn_cartpole.pth"):
        """Loads the policy network's state_dict."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Also update target net
        print(f"Model loaded from {path}")

    def set_eval_mode(self):
        """Sets both networks to evaluation mode."""
        self.policy_net.eval()
        self.target_net.eval()
        self.train = False

    def set_train_mode(self):
        """Sets policy network to train mode (target_net stays in eval)."""
        self.policy_net.train()   
        self.train = True


class Config:
    def __init__(self, n_states, n_actions, lr, memory_capacity, epsilon, gamma, target_update, max_episodes, max_steps, epsilon_min, epsilon_decay, sample_batch_size, use_noisy, play, train):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.memory_capacity = memory_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_update = target_update
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.sample_batch_size = sample_batch_size
        self.use_noisy = use_noisy
        self.play = play
        self.train = train

# 初始化环境
env = gym.make('CartPole-v0') 
env.seed(1) # 设置env随机种子
n_states = env.observation_space.shape[0] # 获取状态的维数
n_actions = env.action_space.n # 获取总的动作数

# 参数设置
cfg = Config(
    n_states=n_states,
    n_actions=n_actions,
    lr = 9e-5,
    memory_capacity=10000,
    sample_batch_size=64,
    epsilon=1.0,
    gamma=0.99,
    target_update=10,
    max_episodes=2000,
    max_steps=1000,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    use_noisy=True,
    play=False,
    train=True
)

agent = DDQN(cfg)

if cfg.train:
    # 初始化wandb
    wandb.init(
        project="easyrl",
        config=vars(cfg)
    )

    rewards = [] # 记录总的rewards
    moving_average_rewards = [] # 记录总的经滑动平均处理后的rewards
    ep_steps = []
    for i_episode in range(1, cfg.max_episodes+1): # cfg.max_episodes为最大训练的episode数
        state = env.reset() # reset环境状态
        ep_reward = 0
        agent.set_train_mode()

        # Sample noise
        agent.policy_net.sample_noise()
        agent.target_net.sample_noise()

        for i_step in range(1, cfg.max_steps+1): # cfg.max_steps为每个episode的补偿
            action = agent.select_action(state) # 根据当前环境state选择action
            next_state, reward, done, _ = env.step(action) # 更新环境参数
            ep_reward += reward
            agent.memory.append((state, action, reward, next_state, done)) # 将state等这些transition存入memory
            state = next_state # 跳转到下一个状态
            agent.update() # 每步更新网络
            if done:
                break
        # 更新target network，复制DQN中的所有weights and biases
        if i_episode % cfg.target_update == 0: #  cfg.target_update为target_net的更新频率
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print('Episode:', i_episode, ' Reward: %i' %
            int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % agent.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)
        
        if cfg.use_noisy:
            wandb.log({
                "Episode Reward": ep_reward,
                "Moving Average Reward": moving_average_rewards,
                "Episode Steps": i_step,
                "Episode": i_episode,
                "Total Training Steps": agent.step
            })
        else:
            wandb.log({
                "Episode Reward": ep_reward,
                "Moving Average Reward": moving_average_rewards,
                "Episode Steps": i_step,
                "Epsilon": agent.config.epsilon,
                "Episode": i_episode,
                "Total Training Steps": agent.step
            })

    # Save the model to the disk
    agent.save_model()

else:
    agent.load_model()

# Set the agent to eval mode
agent.set_eval_mode()

if cfg.play:
    print("\n--- Playing the game with the trained agent ---")
    num_test_episodes = 1  # Number of episodes to play
    for i_ep in range(num_test_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        print(f"\nPlaying Episode {i_ep + 1}")
        env.render() # Display the initial state of the environment
        while not done:
            action = agent.select_action(state)  
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            env.render() # Display the environment after each step
            import time
            time.sleep(0.1) 
        print(f"Episode {i_ep + 1} finished with reward: {ep_reward}")

    env.close()
