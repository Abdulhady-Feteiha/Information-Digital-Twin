import time
import gym
from env.new_env import new_env
from env.new_env2 import new_env2
from env.big_env import big_env
from env.hor_env import hor_env
from env.hor_env2 import hor_env2
original_env = gym.make("Taxi-v3", render_mode="human").env
# original_env = gym.make("Taxi-v3", render_mode="rgb_array").env

env = original_env
training_episodes = 10000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.
train_flag = False
# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Chance of selecting a random action instead of maximising reward.
early_stop_condition = 10
q_table_DIR = "Q_tables/original_env.npy"
results_DIR = f"results/original_env.xlsx"