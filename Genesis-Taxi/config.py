import time
import gym
from env.new_env import new_env
from env.new_env2 import new_env2
from env.big_env import big_env
from env.hor_env import hor_env
from env.hor_env2 import hor_env2
from env.original_env import original_env
from env.env import env
from env import config_general
import numpy as np
# original_env = gym.make("Taxi-v3", render_mode="human").env
# original_env = gym.make("Taxi-v3", render_mode="rgb_array").env

env = env
training_episodes = 10000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.
test_episodes = 1000
train_flag = False
display_flag = True
# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.8 # Discount Rate
epsilon = 0.8 # Chance of selecting a random action instead of maximising reward.
early_stop_condition = 10
#approach to use 
#values are 'normal','one','two','three'
approach = config_general.approach

q_table_DIR = f"Q_tables/eps_{epsilon}_{approach}_gamma_{gamma}.npy"
results_DIR = f"results/eps_{epsilon}_{approach}_gamma_{gamma}.xlsx"
report_DIR = f"reports/eps_{epsilon}_{approach}_gamma_{gamma}.xlsx"