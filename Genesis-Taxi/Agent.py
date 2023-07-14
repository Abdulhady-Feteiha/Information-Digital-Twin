import gym
import numpy as np
import random
from time import sleep
import matplotlib.pyplot as plt
from scipy.special import entr
from utils import clear,calculate_entropy

import config

class Agent():
    def __init__(self):
        clear()
        """Setup"""

        # env = gym.make("Taxi-v3", render_mode="human").env # Setup the Gym Environment
        self.env = config.env # Setup the Gym Environment
        # env = TaxiEnvCustomized(render_mode='human')
        # self.env = TaxiEnvCustomized(render_mode='rgb_array')

    def make(self):
        # Make a new matrix filled with zeros.
        # The matrix will be 500x6 as there are 500 states and 6 actions.
        if self.train:
            q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        else:
            q_table = np.load(config.q_table_DIR)

        # For plotting metrics
        self.all_epochs = []
        self.all_penalties = []
        return q_table
    def train(self,q_table):
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
            if i%100==0:
                print("episode: ",i)
            state = self.env.reset()[0] # Reset returns observation state and other info. We only need the state.
            done = False
            penalties, reward, = 0, 0
            num_steps = 0
            rewards = []
            entropy_value = 0
            while not done:
                num_steps+=1
                if random.uniform(0, 1) < config.epsilon:
                    action = self.env.action_space.sample() # Pick a new action for this state.
                else:
                    action = np.argmax(q_table[state]) # Pick the action which has previously given the highest reward.

                next_state, reward, done, truncated,info = self.env.step(action) 
                rewards.append(reward)
                old_value = q_table[state, action] # Retrieve old value from the q-table.
                next_max = np.max(q_table[next_state])

                # Update q-value for current state.
                new_value = (1 - config.alpha) * old_value + config.alpha * (reward + config.gamma * next_max)
                q_table[state, action] = new_value

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
            episodes_info_gain.append(calculate_entropy(q_table)-past_intropy)
            episodes_entropy.append(calculate_entropy(q_table))
            episodes_penalty.append(penalties)
            # if episodes_info_gain[-1]<0.01:
            #     print("early stopping")
            #     return q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain

        print("Training finished.\n")
        return q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain




# """Display and evaluate agent's performance after Q-learning."""

# total_epochs, total_penalties = 0, 0

# # for _ in range(display_episodes):
# #     state,info_ = env.reset()
# #     epochs, penalties, reward = 0, 0, 0
    
# #     done = False
    
# #     while not done:
# #         action = np.argmax(q_table[state])
# #         state, reward, done, truncated,info = env.step(action)

# #         if reward == -10:
# #             penalties += 1

# #         epochs += 1
# #         clear()
# #         env.render()
# #         print(f"Timestep: {epochs}")
# #         print(f"State: {state}")
# #         print(f"Action: {action}")
# #         print(f"Reward: {reward}")
# #         sleep(0.15) # Sleep so the user can see the 

# #     total_penalties += penalties
# #     total_epochs += epochs

# # print(f"Results after {display_episodes} episodes:")
# # print(f"Average timesteps per episode: {total_epochs / display_episodes}")
# # print(f"Average penalties per episode: {total_penalties / display_episodes}")
