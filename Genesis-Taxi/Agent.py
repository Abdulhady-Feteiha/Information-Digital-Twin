import gym
import numpy as np
import random
from time import sleep
import matplotlib.pyplot as plt
from scipy.special import entr
from utils import clear,calculate_entropy,save_training_progress,early_stop
import config
import time
import cv2
class Agent():
    def __init__(self):
        clear()
        """Setup"""

        # env = gym.make("Taxi-v3", render_mode="human").env # Setup the Gym Environment
        if config.train_flag:
            # self.env = config.env(render_mode='rgb_array') # Setup the Gym Environment
            self.env = config.env
        else:
            # self.env = config.env(render_mode='human') # Setup the Gym Environment
            self.env = config.env
        self.train_flag = config.train_flag
        # env = TaxiEnvCustomized(render_mode='human')
        # self.env = TaxiEnvCustomized(render_mode='rgb_array')
        if self.train_flag:
            self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
            # self.q_table = np.random.rand(self.env.observation_space.n, self.env.action_space.n)
        else:
            self.q_table = np.load(config.q_table_DIR)

        # For plotting metrics
        self.all_epochs = []
        self.all_penalties = []

    def train(self):
        """Training the Agent"""

        # reward_window = []
        # entropies = []
        episodes_num_steps = []
        epsiodes_cumulative_reward = []
        epsiodes_mean_reward = []
        episodes_entropy = []
        episodes_penalty = []
        episodes_info_gain = []
        for i in range(config.training_episodes):
            t0 = time.time()
            if i%100==0:
                print("episode: ",i)
                save_training_progress(self.q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain)
            state = self.env.reset()[0] # Reset returns observation state and other info. We only need the state.
            done = False
            penalties, reward = 0, 0
            num_steps = 0
            rewards = []
            entropy_value = 0
            print(i)
            while not done:
                num_steps+=1
                if random.uniform(0, 1) < config.epsilon:
                    # action = self.env.action_space.sample() # Pick a new action for this state.
                    action = self.env.action_space.sample(info["action_mask"])
                else:
                    action = np.argmax(self.q_table[state]) # Pick the action which has previously given the highest reward.

                next_state, reward, done, truncated,info = self.env.step(action) 
                rewards.append(reward)
                old_value = self.q_table[state, action] # Retrieve old value from the q-table.
                next_max = np.max(self.q_table[next_state])

                # Update q-value for current state.
                alpha_factor = np.log(np.sum(info["action_mask"])/334)
                # print(alpha_factor)
                new_value = (1 - config.alpha) * old_value + config.alpha * (alpha_factor+reward + config.gamma * next_max)
                # print(new_value)
                self.q_table[state, action] = new_value

                if reward == -10: # Checks if agent attempted to do an illegal action.
                    penalties += 1

                state = next_state
            episodes_num_steps.append(num_steps)
            epsiodes_cumulative_reward.append(np.sum(rewards))
            epsiodes_mean_reward.append(np.average(rewards))
            if i==0:
                past_intropy=0
            else:
                past_intropy = episodes_entropy[-1]
            t1 =time.time()

            # entropy = calculate_entropy(self.q_table)[0]
            entropy = 0
            episodes_info_gain.append(entropy-past_intropy)
            episodes_entropy.append(entropy)
            episodes_penalty.append(penalties)
            # print(time.time()-t0)
            # print(time.time()-t1)
            if early_stop(epsiodes_cumulative_reward):
                print(f"early stopped training at episode: {i}")
                return self.q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain
            # if episodes_info_gain[-1]<0.01:
            #     print("early stopping")
            #     return self.q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain

        print("Training finished.\n")
        return self.q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain




    """Display and evaluate agent's performance after Q-learning."""
    def display(self):
        total_epochs, total_penalties = 0, 0

        for _ in range(config.display_episodes):
            state,info_ = self.env.reset()
            epochs, penalties, reward = 0, 0, 0
            
            done = False
            
            while not done:
                action = np.argmax(self.q_table[state])
                print(self.q_table[state])
                state, reward, done, truncated,info = self.env.step(action)
                print(info["action_mask"])

                if reward == -10:
                    penalties += 1

                epochs += 1
                # clear()
                self.env.render()
                print(f"Timestep: {epochs}")
                print(f"State: {state}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                sleep(0.15) # Sleep so the user can see the 

            total_penalties += penalties
            total_epochs += epochs

        print(f"Results after {config.display_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / config.display_episodes}")
        print(f"Average penalties per episode: {total_penalties / config.display_episodes}")
