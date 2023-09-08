from Agent import Agent
from utils import save_training_progress,evaluate
import config
if __name__ == '__main__':

    agent = Agent()

    if config.train_flag:
        q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain = agent.train()
        save_training_progress(q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain)
    else:
        agent.test()
    if config.display_flag:
        agent.display()
    
    # evaluate()