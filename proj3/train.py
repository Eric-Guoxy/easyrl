import gym
import wandb
import numpy as np
from pendulum_v1 import DDPG, Config

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

            # 更新target networks，复制DDPG中的所有weights and biases
            if i_episode % self.config.target_q_update == 0: 
                self.agent.target_q_net.load_state_dict(self.agent.q_net.state_dict())
            if i_episode % self.config.target_actor_update == 0:
                self.agent.target_actor_net.load_state_dict(self.agent.actor_net.state_dict())
            print('Episode:', i_episode, ' Reward: %i' %
                int(ep_reward), 'n_steps:', i_step, 'done: ', done)
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
            "Moving Average Reward": moving_average_rewards,
            "Episode Steps": i_step,
            "Episode": i_episode,
        })

env = gym.make('Pendulum-v1') 
env.seed(1)
n_states = env.observation_space.shape[0] # 获取状态的维数(4)

ddpg_config = Config(
    n_states=n_states,
    q_lr = 9e-5,
    actor_lr=9e-5,
    memory_capacity=10000,
    sample_batch_size=64,
    gamma=0.99,
    tau=0.05,
    target_q_update=10,
    target_actor_update=10,
    max_episodes=2000,
    max_steps=200,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    use_noisy=True,
    play=False,
    train=True
)

ALGORITHM = "DDPG" # ["DDPG", "TD3", "PPO"]
PLAY_ROUNDS = 1

if ALGORITHM == "DDPG":
    trainer = DDPGTrainer(config=ddpg_config, env=env)
    config = ddpg_config
    agent = DDPG(ddpg_config)

if config.train:
    trainer.train()
    agent = trainer.agent()
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
            action = agent.select_action(state)  
            next_state, reward, done, _ = env.step(action)
            state = next_state
            ep_reward += reward
            env.render() # Display the environment after each step
            import time
            time.sleep(0.1) 
        print(f"Episode {i_ep + 1} finished with reward: {ep_reward}")

    env.close()

