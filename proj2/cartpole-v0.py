# This file implements the CartPole-v0 classic rl example.
# This program will use DDQN as the backbone RL algorithm.
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque 
import random
import matplotlib.pyplot as plt

class QNet(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Store parameters
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Used layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Used non-linear function
        self.gelu = nn.GELU()

    def forward(self, x):
        # The input layer
        x = self.input_layer(x)

        # The hidden layers
        for _ in range(self.num_layers - 1):
            x = self.hidden_layer(x)
            x = self.gelu(x)
        
        # The final layer
        x = self.output_layer(x)

        return x


class DDQN:
    def __init__(self, config):
        # Set up the two nets
        self.target_net = QNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=4*config.n_states,
            output_dim=config.n_actions
        )
        self.target_net.eval()

        self.policy_net = QNet(
            num_layers=3,
            input_dim=config.n_states,
            hidden_dim=4*config.n_states,
            output_dim=config.n_actions
        )

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
        # Using epsilon-greedy algorithm
        if random.random() < self.config.epsilon and self.train:
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

        # Convert to Pytorch tensors
        batch_state = torch.FloatTensor(np.array(batch_state))
        batch_action = torch.LongTensor(np.array(batch_action)).unsqueeze(1) # Change shape [2] to [2, 1]
        batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state))
        batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1)

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

        if self.config.epsilon > self.config.epsilon_min:
            self.config.epsilon *= self.config.epsilon_decay

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
    def __init__(self, n_states, n_actions, lr, memory_capacity, epsilon, gamma, target_update, max_episodes, max_steps, epsilon_min, epsilon_decay, sample_batch_size):
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

# 初始化环境
env = gym.make('CartPole-v0') 
env.seed(1) # 设置env随机种子
n_states = env.observation_space.shape[0] # 获取总的状态数
n_actions = env.action_space.n # 获取总的动作数

train = False
play = True

# 参数设置
cfg = Config(
    n_states=n_states,
    n_actions=n_actions,
    lr = 0.001,
    memory_capacity=10000,
    sample_batch_size=64,
    epsilon=1.0,
    gamma=0.99,
    target_update=10,
    max_episodes=1000,
    max_steps=1000,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    
)

agent = DDQN(cfg)

if train:
    rewards = [] # 记录总的rewards
    moving_average_rewards = [] # 记录总的经滑动平均处理后的rewards
    ep_steps = []
    for i_episode in range(1, cfg.max_episodes+1): # cfg.max_episodes为最大训练的episode数
        state = env.reset() # reset环境状态
        ep_reward = 0
        agent.set_train_mode()
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
            int(ep_reward), 'n_steps:', i_step, 'done: ', done,' Explore: %.2f' % cfg.epsilon)
        ep_steps.append(i_step)
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if i_episode == 1:
            moving_average_rewards.append(ep_reward)
        else:
            moving_average_rewards.append(
                0.9*moving_average_rewards[-1]+0.1*ep_reward)

    print("--- The visualization of the training rewards and the moving average version ---")
    # Plot 1: Rewards per episode
    plt.figure(figsize=(10, 6)) # Create the first figure
    plt.plot(
        rewards,
        label="Rewards"
    )
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend() 
    plt.grid(True) 
    plt.savefig("rewards_per_episode.png")

    # Plot 2: Moving Average Rewards
    plt.figure(figsize=(10, 6)) # Create the second figure
    plt.plot(
        moving_average_rewards,
        label="Moving average rewards",
        color='green'
    )
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward")
    plt.title("Moving Average of Rewards")
    plt.legend() 
    plt.grid(True) 
    plt.savefig("moving_average_rewards.png")

    # Save the model to the disk
    agent.save_model()

else:
    agent.load_model()

# Set the agent to eval mode
agent.set_eval_mode()

if play:
    print("\n--- Playing the game with the trained agent ---")
    num_test_episodes = 5  # Number of episodes to play
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
