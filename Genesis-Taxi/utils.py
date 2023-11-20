# Define function to clear console window.
import numpy as np
from sklearn import preprocessing as pre
import os
from os import system, name
import pandas as pd
import config
import matplotlib.pyplot as plt
def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

def calculate_entropy(q_table):
    """
    Calculates the entropy of a Q table.

    Args:
    q_table: A NumPy array representing the Q table.

    Returns:
    The entropy of the Q table.
    """
    total_entropy = 0
    for state in q_table:
        state = state.reshape(-1, 1)
        entropies = pre.MinMaxScaler().fit_transform(state)
        total_entropy+=sum(entropies)
#         print("q values:",state)
        # print("entropies",entropies)

    return total_entropy

# Calculate the entropy of the Q table.
def save_training_progress(q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain):
    # results_df = pd.DataFrame(columns = ['episode','num_steps','penalty','cumlative_reward','Entropy','Sucess_rate','Robustness'])
    columns = ['episode','num_steps','penalty','cumlative_reward','Entropy','Sucess_rate','Robustness']
    np.save(config.q_table_DIR,q_table)
    episodes = [ i for i in range(len(epsiodes_cumulative_reward))]
    # for i in range(len(episodes_num_steps)):
    results_df = pd.DataFrame({"episode":episodes,'num_steps':episodes_num_steps,"mean_reward":epsiodes_mean_reward,"cumulative_reward":epsiodes_cumulative_reward,
                                    "entropy":episodes_entropy,"penalty":episodes_penalty,"info_gain":episodes_info_gain}
                                    )
    results_df.to_excel(config.results_DIR)

def report(episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_penalty):
    episodes = [ i for i in range(len(epsiodes_cumulative_reward))]
    # for i in range(len(episodes_num_steps)):
    results_df = pd.DataFrame({"episode":episodes,'num_steps':episodes_num_steps,"mean_reward":epsiodes_mean_reward,"cumulative_reward":epsiodes_cumulative_reward,"penalty":episodes_penalty}
                                    )
    results_df.to_excel(config.report_DIR)

def early_stop(epsiodes_cumulative_reward):
    if np.average(epsiodes_cumulative_reward[-10:])>config.early_stop_condition:
        return True
    else:
        return False
def evaluate():
    results_df = pd.read_excel(config.results_DIR)
    # print(results_df.loc[:,"episode"])
    # print(results_df.loc[:,"num_steps"])
    figure, axis = plt.subplots(3, 2)
    plt.plot(results_df.loc[:,"episode"],results_df.loc[:,"num_steps"])
    axis[0, 0].plot(results_df.loc[:,"episode"],results_df.loc[:,"num_steps"])
    axis[0, 0].set_title("numb steps")
    axis[0, 1].plot(results_df.loc[:,"episode"],results_df.loc[:,"mean_reward"])
    axis[0, 1].set_title("mean reward")
    axis[1, 0].plot(results_df.loc[:,"episode"],results_df.loc[:,"cumulative_reward"])
    axis[1, 0].set_title("cumulative reward")
    axis[1, 1].plot(results_df.loc[:,"episode"],results_df.loc[:,"entropy"])
    axis[1, 1].set_title("entropy")
    axis[2, 0].plot(results_df.loc[:,"episode"],results_df.loc[:,"penalty"])
    axis[2, 0].set_title("penalties")
    axis[2, 1].plot(results_df.loc[:,"episode"],results_df.loc[:,"info_gain"])
    axis[2, 1].set_title("information gain")
    # ax1 = results_df.plot.scatter(x='episode',y='num_steps',c='DarkBlue')
    plt.show()
