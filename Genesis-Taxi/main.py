from Agent import Agent
from utils import save_training_progress,evaluate

agent = Agent()
q_table = agent.make()
q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain = agent.train(q_table)
save_training_progress(q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain)
evaluate()