import gym
import numpy as np
import random
from time import sleep
import matplotlib.pyplot as plt
from scipy.special import entr
from utils import clear,calculate_entropy,save_training_progress,early_stop,report
import config
import time
import cv2
class Agent():
    def __init__(self):
        clear()
        """Setup"""
        # env = gym.make("Taxi-v3", render_mode="human").env # Setup the Gym Environment
        if config.display_flag:
            self.env = config.env(render_mode='human') # Setup the Gym Environment
            # self.env = config.env
        else:
            self.env = config.env(render_mode='rgb_array') # Setup the Gym Environment
            # self.env = config.env
        self.train_flag = config.train_flag
        self.matrix = config.matrix
        # env = TaxiEnvCustomized(render_mode='human')
        # self.env = TaxiEnvCustomized(render_mode='rgb_array')
        self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        if self.train_flag:
            if config.approach == 'normal' or config.approach == 'two':
                self.q_table = np.zeros([self.env.observation_space.n, self.env.action_space.n])
            else:
                self.q_table = self.calculate_q_table(self.matrix)
            # self.q_table = np.random.rand(self.env.observation_space.n, self.env.action_space.n)
        else:
            self.q_table = np.load(config.q_table_DIR)

        # For plotting metrics
        self.all_epochs = []
        self.all_penalties = []

    def calculate_q_table(self,matrix):
        """ Intitalize the Q table and do necessary preprocessing """
        q_table = self.q_table
        no_of_states = self.env.observation_space.n
        no_of_pass_locations = 5
        no_of_dest_locations = 4
        no_of_grids = int(no_of_states) / (no_of_pass_locations*no_of_dest_locations)
        no_of_rows = no_of_cols = int(np.sqrt(no_of_grids))
        no_of_actions = self.env.action_space.n
        for row in range(no_of_rows):
            for col in range(no_of_cols):
                for pass_idx in range(no_of_pass_locations):
                    for dest_idx in range(no_of_dest_locations):
                        state = self.env.encode(row, col, pass_idx, dest_idx)
                        #print(self.q_table[state])
                        for action in range(no_of_actions):
                            try:
                                if action == 0:
                                    q_table[state,action] = self.matrix[row][col]-self.matrix[row+1][col]
                                if action == 1:
                                    q_table[state,action] = self.matrix[row][col]-self.matrix[row-1][col]
                                if action == 2:
                                    q_table[state,action] = self.matrix[row][col]-self.matrix[row][col+1]
                                if action == 3:
                                    q_table[state,action] = self.matrix[row][col]-self.matrix[row][col-1]
                                if action == 4 or action==5:
                                    if dest_idx==pass_idx:
                                        q_table[state,action] = 0
                                    else:
                                        q_table[state,action] = config.illegal_pen

                            except Exception as e:
                                q_table[state,action] = config.illegal_pen
                                
        return q_table

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
            print(i)

            if i%100==0:
                print("episode: ",i)
                save_training_progress(self.q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain)
            state = self.env.reset()[0] # Reset returns observation state and other info. We only need the state.
            
            done = False
            penalties, reward = 0, 0
            num_steps = 0
            rewards = []
            entropy_value = 0
            #print(i)
            while not done:
                num_steps+=1
                if random.uniform(0, 1) < config.epsilon:
                    action = self.env.action_space.sample() # Pick a new action for this state.
                    #action = self.env.action_space.sample(info["action_mask"])
                else:
                    action = np.argmax(self.q_table[state]) # Pick the action which has previously given the highest reward.

                next_state, reward, done, truncated,info = self.env.step(action)

                rewards.append(reward)
                old_value = self.q_table[state, action] # Retrieve old value from the q-table.
                next_max = np.max(self.q_table[next_state])

                if config.approach == 'normal' or config.approach == 'one':
                    new_value = (1 - config.alpha) * old_value + config.alpha * (reward + config.gamma * next_max)
                else:
                    row,col,_,_ = self.env.decode(state)
                    next_row,next_col,_,_ = self.env.decode(next_state)
                    # print("state: ",row,col)
                    # print("next state: ",next_row,next_col)
                    # print("action: ",action)
                    # print(self.q_table[state])
                    alpha_old = self.matrix[row][col]
                    alpha_new = self.matrix[next_row][next_col]
                    alpha_difference = alpha_new - alpha_old

                    new_value = (1 - config.alpha) * old_value + config.alpha * ((reward+alpha_difference) + config.gamma * next_max)
                #update tue alpha change
                
                # Update q-value for current state.
                #alpha_factor = np.log(np.sum(info["action_mask"])/334)
                # print(alpha_factor)
                #new_value = (1 - config.alpha) * old_value + config.alpha * (alpha_factor+reward + config.gamma * next_max)
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
                action = 4
                print("Q table of state",self.q_table[state])
                print(f"P: {self.env.P[state][action]}")

                state, reward, done, truncated,info = self.env.step(action)
                print("action mask",info["action_mask"])

                if reward == -10:
                    penalties += 1

                epochs += 1
                # clear()
                self.env.render()
                print(f"Timestep: {epochs}")
                print(f"State: {state}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                sleep(20) # Sleep so the user can see the 

            total_penalties += penalties
            total_epochs += epochs

        print(f"Results after {config.display_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / config.display_episodes}")
        print(f"Average penalties per episode: {total_penalties / config.display_episodes}")

    def test(self):
        fail_count, total_epochs, total_penalties = 0, 0, 0
        episodes_num_steps = []
        epsiodes_cumulative_reward = []
        epsiodes_mean_reward = []
        episodes_entropy = []
        episodes_penalty = []
        episodes_info_gain = []
        for i in range(config.test_episodes):
            state,info_ = self.env.reset(seed=i)
            epochs, penalties, reward = 0, 0, 0
            num_steps = 0
            rewards = []
            entropy_value = 0
            done = False
            print(i)
            while not done:
                num_steps+=1
                action = np.argmax(self.q_table[state])
                # print(self.q_table[state])
                state, reward, done, truncated,info = self.env.step(action)
                # print(info["action_mask"])
                rewards.append(reward)

                if reward == -10:
                    penalties += 1

                epochs += 1
                # clear()
                self.env.render()
                # print(f"Timestep: {epochs}")
                # print(f"State: {state}")
                # print(f"Action: {action}")
                # print(f"Reward: {reward}")
                # sleep(0.15) # Sleep so the user can see the 
                if num_steps>100:
                    fail_count+=1
                    break
            total_penalties += penalties
            total_epochs += epochs
            episodes_num_steps.append(num_steps)
            epsiodes_cumulative_reward.append(np.sum(rewards))
            epsiodes_mean_reward.append(np.average(rewards))
            entropy = calculate_entropy(self.q_table)[0]
            # episodes_info_gain.append(entropy-past_intropy)
            # episodes_entropy.append(entropy)
            episodes_penalty.append(penalties)
        report(episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_penalty)
        print(f"Results after {config.display_episodes} episodes:")
        print(f"Average timesteps per episode: {total_epochs / config.display_episodes}")
        print(f"Average penalties per episode: {total_penalties / config.display_episodes}")
        print("Entropy of the Q table",entropy)
        print("Sucess rate:",(config.test_episodes-fail_count)/config.test_episodes)
