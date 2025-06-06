import gym
import wandb
import numpy as np
from pendulum_v1 import DDPG, TD3, DDPGConfig, TD3Config

class DDPGTrainer:
    def __init__(self, config, env):
        self.config = config 
        self.env = env
        self.agent = DDPG(config)

        # Init wandb
        wandb.init(
            project="easyrl",
            config=vars(config),
            name="ddpg-pendulum"
        )

    def train(self):
        env = self.env
        rewards = [] # 记录总的rewards
        moving_average_rewards = [] # 记录总的经滑动平均处理后的rewards
        ep_steps = []
        for i_episode in range(1, self.config.max_episodes+1): # cfg.max_episodes为最大训练的episode数
            # 添加噪声
            self.agent.target_q_net.sample_noise()
            self.agent.q_net.sample_noise()

            state = env.reset() # reset环境状态
            ep_reward = 0
            for i_step in range(1, self.config.max_steps+1): # cfg.max_steps为每个episode的补偿
                action = self.agent.select_action(state) # 根据当前环境state选择action
                next_state, reward, done, _ = env.step(np.array([action])) # 更新环境参数
                ep_reward += reward
                self.agent.memory.append((state, action, reward, next_state, done)) # 将state等这些transition存入memory
                state = next_state # 跳转到下一个状态
                self.agent.update() # 每步更新网络
                if done:
                    break

            print('Episode:', i_episode, ' Reward: ',
                   ep_reward, 'n_steps:', i_step, 'done: ', done)
            ep_steps.append(i_step)
            rewards.append(ep_reward)
            # 计算滑动窗口的reward
            if i_episode == 1:
                moving_average_rewards.append(ep_reward)
            else:
                moving_average_rewards.append(
                    0.9*moving_average_rewards[-1]+0.1*ep_reward)
        
            # Log following metrics
            wandb.log({
                "Episode Reward": ep_reward,
                "Moving Average Reward": moving_average_rewards[-1],
                "Episode Steps": i_step,
                "Episode": i_episode,
            })


class TD3Trainer:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.agent = TD3(config)

        # Init wandb
        wandb.init(
            project="easyrl",
            config=vars(config),
            name="td3-pendulum"
        )

    def train(self):
        env = self.env
        rewards = [] # 记录总的rewards
        moving_average_rewards = [] # 记录总的经滑动平均处理后的rewards
        ep_steps = []
        for i_episode in range(1, self.config.max_episodes+1): # cfg.max_episodes为最大训练的episode数
            # 添加噪声
            self.agent.target_q_net_1.sample_noise()
            self.agent.q_net_1.sample_noise()
            self.agent.target_q_net_2.sample_noise()
            self.agent.q_net_2.sample_noise()

            state = env.reset() # reset环境状态
            ep_reward = 0
            for i_step in range(1, self.config.max_steps+1): # cfg.max_steps为每个episode的补偿
                action = self.agent.select_action(state) # 根据当前环境state选择action
                next_state, reward, done, _ = env.step(np.array([action])) # 更新环境参数
                ep_reward += reward
                self.agent.memory.append((state, action, reward, next_state, done)) # 将state等这些transition存入memory
                state = next_state # 跳转到下一个状态
                self.agent.update() # 每步更新网络
                if done:
                    break

            print('Episode:', i_episode, ' Reward: ', 
                ep_reward, 'n_steps:', i_step, 'done: ', done)
            ep_steps.append(i_step)
            rewards.append(ep_reward)
            # 计算滑动窗口的reward
            if i_episode == 1:
                moving_average_rewards.append(ep_reward)
            else:
                moving_average_rewards.append(
                    0.9*moving_average_rewards[-1]+0.1*ep_reward)
        
            # Log following metrics
            wandb.log({
                "Episode Reward": ep_reward,
                "Moving Average Reward": moving_average_rewards[-1],
                "Episode Steps": i_step,
                "Episode": i_episode,
            })

env = gym.make('Pendulum-v1') 
env.seed(1)
n_states = env.observation_space.shape[0] # 获取状态的维数(4)

ddpg_config = DDPGConfig(
    n_states=n_states,
    q_lr = 1e-3,
    actor_lr=3e-4,
    memory_capacity=10000,
    sample_batch_size=64,
    gamma=0.99,
    tau=0.005,
    max_episodes=2000,
    max_steps=200,
    use_noisy=True,
    play=False,
    train=True,
    action_max=2.0,
    action_min=-2.0,
    action_noise_std=0.1
)

td3_config = TD3Config(
    n_states=n_states,
    q_lr = 1e-3,
    actor_lr=3e-4,
    memory_capacity=10000,
    sample_batch_size=64,
    gamma=0.99,
    tau=0.005,
    max_episodes=2000,
    max_steps=200,
    use_noisy=True,
    play=False,
    train=True,
    action_max=2.0,
    action_min=-2.0,
    action_noise_std=0.1,
    policy_noise_std=0.2,
    policy_delay=2,
    noise_clip=0.5
)

ALGORITHM = "TD3" # ["DDPG", "TD3", "PPO", "GRPO"]
PLAY_ROUNDS = 1

if ALGORITHM == "DDPG":
    config = ddpg_config
    trainer = DDPGTrainer(config=config, env=env)
    agent = trainer.agent
elif ALGORITHM == "TD3":
    config = td3_config
    trainer = TD3Trainer(config=config, env=env)
    agent = trainer.agent

if config.train:
    trainer.train()
    trainer.agent.save_model()
else:
    agent.load_model()

if config.play:
    agent.set_eval_mode()

    print("\n--- Playing the game with the trained agent ---")
    num_test_episodes = PLAY_ROUNDS # Number of episodes to play
    for i_ep in range(num_test_episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        print(f"\nPlaying Episode {i_ep + 1}")
        env.render() # Display the initial state of the environment
        while not done:
            action_scaler = agent.select_action(state)  
            next_state, reward, done, _ = env.step(np.array([action_scaler]))
            state = next_state
            ep_reward += reward
            env.render() # Display the environment after each step
            import time
            time.sleep(0.1) 
        print(f"Episode {i_ep + 1} finished with reward: {ep_reward}")

    env.close()

