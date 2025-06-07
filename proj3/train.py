import gym
import wandb
import numpy as np
import torch
from pendulum_v1 import DDPG, TD3, PPO, DDPGConfig, TD3Config, PPOConfig

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


class PPOTrainer:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.agent = PPO(config)

        # Init wandb
        wandb.init(
            project="easyrl",
            config=vars(config),
            name="ppo-pendulum" + wandb.util.generate_id()
        )
    
    def train(self):
        env = self.env
        
        # For overall episodic tracking
        all_episode_rewards = []
        moving_average_rewards = []
        
        current_state = env.reset()
        total_timesteps_processed = 0
        
        # For accumulating rewards and steps within the current episode
        current_episode_reward = 0
        current_episode_steps = 0
        episode_count = 0

        # Determine the number of PPO update cycles
        # Example: run for a total number of steps
        num_update_cycles = self.config.total_steps // self.config.num_steps_per_rollout

        for i_update_cycle in range(1, num_update_cycles + 1):
            # --- Rollout Collection Phase ---
            for _ in range(self.config.num_steps_per_rollout):
                total_timesteps_processed += 1
                current_episode_steps += 1

                action_scalar, logp_scalar = self.agent.select_action(current_state)
                action_to_env = np.array([action_scalar]) # Pendulum expects a 1-element array

                # Get V(s) for the current state before stepping
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.agent.device)
                with torch.no_grad():
                    value_pred = self.agent.critic_net(state_tensor).cpu().item()
                
                next_state, reward, done, _ = env.step(action_to_env)
                current_episode_reward += reward
                
                # Store (s, a, old_logp, V(s), r, done)
                # Note: action_scalar is fine if PPO.update expects scalar action for 1D case,
                # but storing action_to_env (np.array) is more general.
                self.agent.rollout_batch.append((current_state, action_to_env, logp_scalar, value_pred, reward, done))
                
                current_state = next_state

                if done:
                    episode_count += 1
                    all_episode_rewards.append(current_episode_reward)
                    if not moving_average_rewards:
                        moving_average_rewards.append(current_episode_reward)
                    else:
                        moving_average_rewards.append(0.9 * moving_average_rewards[-1] + 0.1 * current_episode_reward)
                    
                    print(f"Update Cycle: {i_update_cycle}, Timestep: {total_timesteps_processed}, Episode: {episode_count} Reward: {current_episode_reward:.2f}, Steps: {current_episode_steps}")
                    
                    # Log episodic metrics immediately when an episode ends
                    wandb.log({
                        "Episodic Reward": current_episode_reward,
                        "Moving Average Reward": moving_average_rewards[-1],
                        "Episode Steps": current_episode_steps,
                        "Total Timesteps": total_timesteps_processed,
                        "Episode Count": episode_count
                    }, step=total_timesteps_processed) # Use total_timesteps_processed as the step for x-axis

                    current_state = env.reset()
                    current_episode_reward = 0
                    current_episode_steps = 0
            
            # --- PPO Update Phase ---
            # The 'current_state' here is S_T (the state after the last action of the rollout)
            # The 'done' status for S_T is the 'done' from the very last transition collected.
            last_done_in_rollout = self.agent.rollout_batch[-1][-1] if self.agent.rollout_batch else False
            
            avg_policy_loss, avg_critic_loss, avg_entropy = self.agent.update(current_state, last_done_in_rollout) # Call update

            # Log metrics related to this PPO update cycle
            # Calculate average reward over the just completed rollout batch for logging
            avg_rollout_reward = np.mean([transition[4] for transition in self.agent.rollout_batch[-self.config.num_steps_per_rollout:]]) if self.agent.rollout_batch else 0

            wandb.log({
                "PPO Update Cycle": i_update_cycle,
                "Average Rollout Reward (per update)": avg_rollout_reward,
                "Policy Loss": avg_policy_loss, 
                "Value Loss": avg_critic_loss,  
                "Entropy": avg_entropy,         
                "Total Timesteps": total_timesteps_processed # Log again to ensure this step has PPO update metrics
            }, step=total_timesteps_processed)
            
            print(f"PPO Update Cycle {i_update_cycle} completed. Avg Rollout Reward: {avg_rollout_reward:.2f}. Total Timesteps: {total_timesteps_processed}")

        print("Training finished.")
        wandb.finish() # Explicitly finish wandb run
            

            
env = gym.make('Pendulum-v1') 
env.seed(1)
n_states = env.observation_space.shape[0] # 获取状态的维数(4)
n_actions = env.action_space.shape[0]

ddpg_config = DDPGConfig(
    n_states=n_states,
    n_actions=n_actions,
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
    n_actions=n_actions,
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

ppo_config = PPOConfig(
    n_states=n_states,
    n_actions=n_actions,
    policy_lr = 1e-3,
    critic_lr=3e-4,
    mini_batch_size=32,
    ppo_epochs=10,
    gamma=0.99,
    use_noisy=False,
    play=False,
    train=True,
    action_max=2.0,
    action_min=-2.0,
    log_std_min=-20,
    log_std_max=2,
    lambda_gae=0.95,
    epsilon=0.2,
    total_steps=10000,
    num_steps_per_rollout=200,
)

ALGORITHM = "PPO" # ["DDPG", "TD3", "PPO", "GRPO"]
PLAY_ROUNDS = 1

if ALGORITHM == "DDPG":
    config = ddpg_config
    trainer = DDPGTrainer(config=config, env=env)
    agent = trainer.agent
elif ALGORITHM == "TD3":
    config = td3_config
    trainer = TD3Trainer(config=config, env=env)
    agent = trainer.agent
elif ALGORITHM == "PPO":
    config = ppo_config
    trainer = PPOTrainer(config=config, env=env)
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

