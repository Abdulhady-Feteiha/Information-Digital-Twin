import time
import gym
from env.new_env import new_env
from env.new_env2 import new_env2
from env.big_env import big_env
from env.hor_env import hor_env
from env.hor_env2 import hor_env2
# original_env = gym.make("Taxi-v3", render_mode="human").env
# original_env = gym.make("Taxi-v3", render_mode="rgb_array").env

env = new_env2
training_episodes = 10000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.
test_episodes = 100
train_flag = False
display_flag = False
# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.8 # Chance of selecting a random action instead of maximising reward.
early_stop_condition = 10
#approach to use 
#values are 'normal','one','two','three'
approach = 'three'
# Heat map values 
illegal_pen = -50

# matrix = [[-6.4429,-7.0279,-7.0279,-6.4429,-6.4429],
#           [-6.4429,-6.4429,-6.4429,-6.0279,-6.4429],
#           [-6.4429,-6.0279,-6.0279,-6.0279,-6.4429],
#           [-7.0279,-6.4429,-6.4429,-6.4429,-6.4429],
#           [-7.0279,-7.0279,-7.0279,-6.4429,-7.0279]]
matrix = [[-6.3866,-6.3866,-6.3866,-6.9715,-6.9715,-6.3866,-6.3866,-6.3866],
          [-6.3866,-5.9715,-5.9715,-6.3866,-6.3866,-5.9715,-5.9715,-6.3866],
          [-6.3866,-5.9715,-5.9715,-6.3866,-6.3866,-5.9715,-5.9715,-6.3866],
          [-6.3866,-5.9715,-5.9715,-6.3866,-6.3866,-5.9715,-5.9715,-6.3866],
          [-6.9715,-6.3866,-5.9715,-5.9715,-5.9715,-6.3866,-6.3866,-6.3866],
          [-6.9715,-6.3866,-5.9715,-5.9715,-5.9715,-6.3866,-6.3866,-6.3866],
          [-6.9715,-6.3866,-6.3866,-6.3866,-5.9715,-6.3866,-6.3866,-6.3866],
          [-6.9715,-6.3866,-6.3866,-6.3866,-5.9715,-6.3866,-5.6496,-6.3866],
          [-6.9715,-6.3866,-6.3866,-6.3866,-5.9715,-6.3866,-6.3866,-6.3866],
          [-6.3866,-6.9715,-6.9715,-6.9715,-6.3866,-6.9715,-6.3866,-6.9715]]
#directories
q_table_DIR = f"Q_tables/new_env2_08_eps_{approach}.npy"
results_DIR = f"results/new_env2_08_eps_{approach}.xlsx"
report_DIR = f"results/report_new_env2_08_eps_{approach}.xlsx"