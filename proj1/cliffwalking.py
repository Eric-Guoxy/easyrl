# This file implements the CliffWalking RL project
import gym
import random
import matplotlib.pyplot as plt
import json

class CliffWalkingWapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
class Config:
    def __init__(self, policy_lr, gamma, train_eps, epsilon):
        self.policy_lr = policy_lr
        self.gamma = gamma
        self.train_eps = train_eps
        self.epsilon = epsilon

class QLearning:
    def __init__(self, env, state_dim, action_dim, learning_rate, gamma, epsilon):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.is_training = True

        for state in range(self.state_dim):
            self.Q[state] = {}
            for action in range(self.action_dim):
                self.Q[state][action] = 0.0
    
    def train(self):
        self.is_training = True
    
    def eval(self):
        self.is_training = False
    
    def choose_action(self, state):
        if self.is_training:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < self.epsilon:
                # Explore
                return self.env.action_space.sample() 
            else:
                # Exploit: choose the action with the highest Q-value for the current state
                return max(self.Q[state], key=self.Q[state].get)
        else:
            # Always choose the best possible action
            return max(self.Q[state], key=self.Q[state].get)
    
    def update(self, state, action, reward, next_state, done):
        if next_state not in self.Q:
            self.Q[next_state] = {act: 0.0 for act in range(self.action_dim)}
        
        if done:
            target = reward
        else:
            max_next_q_value = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0
            target = reward + self.gamma * max_next_q_value
        
        # Calculate the TD error
        td_error = target - self.Q[state][action]

        # Update the Q-value
        self.Q[state][action] += self.learning_rate * td_error
    
    def save_to_disk(self):
        with open("model_q_values.json", "w", encoding="utf-8") as f:
            json.dump(self.Q, f, ensure_ascii=False, indent=4)
        

# Set up the config
cfg = Config(
    policy_lr=0.1,
    gamma=0.9,
    train_eps=500,
    epsilon=0.1
)

'''初始化环境'''  
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
env = CliffWalkingWapper(env)
agent = QLearning(
    env=env,
    state_dim=env.observation_space.n,
    action_dim=env.action_space.n,
    learning_rate=cfg.policy_lr,
    gamma=cfg.gamma,
    epsilon=cfg.epsilon
)

# Set the agent to be the training mode
agent.train()

rewards = []  
ma_rewards = [] # moving average reward
for i_ep in range(cfg.train_eps): # train_eps: 训练的最大episodes数
    ep_reward = 0  # 记录每个episode的reward
    state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    while True:
        action = agent.choose_action(state)  # 根据算法选择一个动作
        next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
        agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
        state = next_state  # 存储上一个观察值
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    if ma_rewards:
        ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
    else:
        ma_rewards.append(ep_reward)
    print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))

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
plt.show() # Show the first plot

# Plot 2: Moving Average Rewards
plt.figure(figsize=(10, 6)) # Create the second figure
plt.plot(
    ma_rewards,
    label="Moving average rewards",
    color='green'
)
plt.xlabel("Episodes")
plt.ylabel("Moving Average Reward")
plt.title("Moving Average of Rewards")
plt.legend() 
plt.grid(True) 
plt.show() 

# Save the model to the disk
agent.save_to_disk()

# Set the agent to eval mode
agent.eval()

print("\n--- Playing the game with the trained agent ---")
num_test_episodes = 5  # Number of episodes to play
for i_ep in range(num_test_episodes):
    state = env.reset()
    ep_reward = 0
    done = False
    print(f"\nPlaying Episode {i_ep + 1}")
    env.render() # Display the initial state of the environment
    while not done:
        action = agent.choose_action(state)  
        next_state, reward, done, _ = env.step(action)
        state = next_state
        ep_reward += reward
        env.render() # Display the environment after each step
        import time
        time.sleep(0.1) 
    print(f"Episode {i_ep + 1} finished with reward: {ep_reward}")

env.close()
