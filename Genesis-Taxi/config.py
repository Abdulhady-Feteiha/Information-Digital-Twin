import time
import gym
from env.new_env import new_env
from env.new_env2 import new_env2
from env.big_env import big_env
from env.hor_env import hor_env
from env.hor_env2 import hor_env2
#original_env = gym.make("Taxi-v3", render_mode="human").env
original_env = gym.make("Taxi-v3", render_mode="rgb_array").env

env = original_env
training_episodes = 10000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.
train_flag = True
# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.8 # Chance of selecting a random action instead of maximising reward.
early_stop_condition = 10
#approach to use 
#values are 'normal','one','two','three'
approach = 'three'
# Heat map values 
# matrix = [[-0.2605,-0.2605,-0.2605,-0.2605,-0.2605],
#           [0.1545,0.1545,0.1545,0.4764,0.1545],
#           [0.1545,0.4764,0.4764,0.4764,0.1545],
#           [-0.2605,0.1545,0.1545,0.1545,0.1545],
#           [-0.8455,-0.2605,-0.2605,-0.2605,-0.2605]]

matrix = [[-4.939,-4.939,-4.939,-4.939,-4.939],
          [-5.524,-5.524,-5.524,-4.202,-5.524],
          [-5.524,-4.202,-4.202,-4.202,-5.524],
          [-4.939,-4.524,-4.524,-4.202,-5.524],
          [-5.524,-4.939,-4.939,-4.939,-4.939]]
#directories
q_table_DIR = f"Q_tables_Amr/original_env_approach_{approach}.npy"
results_DIR = f"results_Amr/original_env_approach_{approach}.xlsx"