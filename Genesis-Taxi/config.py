import time
import gym 
from env.new_env import new_env
from env.new_env2 import new_env2
from env.big_env import big_env
from env.hor_env import hor_env
from env.hor_env2 import hor_env2
from env.original_env import original_env

# original_env = gym.make("Taxi-v3", render_mode="human").env
#original_env = gym.make("Taxi-v3", render_mode="rgb_array").env

env = original_env
training_episodes = 10000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.
test_episodes = 100
train_flag = False
display_flag = True
# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.1 # Chance of selecting a random action instead of maximising reward. #0.1
early_stop_condition = 10
#approach to use 
#values are 'normal','one','two','three'
approach = 'normal'
# Heat map values 
illegal_pen = -50

q_matrix = [[-0.2605,-0.2605,-0.2605,-0.2605,-0.2605],
          [0.1545,0.1545,0.1545,0.4764,0.1545],
          [0.1545,0.4764,0.4764,0.4764,0.1545],
          [-0.2605,0.1545,0.1545,0.1545,0.1545],
          [-0.8455,-0.2605,-0.2605,-0.2605,-0.2605]]

alpha_matrix = [[-4.939,-4.939,-4.939,-4.939,-4.939],
          [-5.524,-5.524,-5.524,-4.202,-5.524],
          [-5.524,-4.202,-4.202,-4.202,-5.524],
          [-4.939,-4.524,-4.524,-4.202,-5.524],
          [-5.524,-4.939,-4.939,-4.939,-4.939]]
# matrix = [[-6.3866,-6.3866,-6.3866,-6.9715,-6.9715,-6.3866,-6.3866,-6.3866],
#           [-6.3866,-5.9715,-5.9715,-6.3866,-6.3866,-5.9715,-5.9715,-6.3866],
#           [-6.3866,-5.9715,-5.9715,-6.3866,-6.3866,-5.9715,-5.9715,-6.3866],
#           [-6.3866,-5.9715,-5.9715,-6.3866,-6.3866,-5.9715,-5.9715,-6.3866],
#           [-6.9715,-6.3866,-5.9715,-5.9715,-5.9715,-6.3866,-6.3866,-6.3866],
#           [-6.9715,-6.3866,-5.9715,-5.9715,-5.9715,-6.3866,-6.3866,-6.3866],
#           [-6.9715,-6.3866,-6.3866,-6.3866,-5.9715,-6.3866,-6.3866,-6.3866],
#           [-6.9715,-6.3866,-6.3866,-6.3866,-5.9715,-6.3866,-5.6496,-6.3866],
#           [-6.9715,-6.3866,-6.3866,-6.3866,-5.9715,-6.3866,-6.3866,-6.3866],
#           [-6.3866,-6.9715,-6.9715,-6.9715,-6.3866,-6.9715,-6.3866,-6.9715]]
#directories
q_table_DIR = f"Q_tables_Amr/original_env_approach_{approach}.npy"
results_DIR = f"results_Amr/original_env_approach_{approach}.xlsx"
report_DIR = f"results/report_original_env_approach_{approach}.xlsx"