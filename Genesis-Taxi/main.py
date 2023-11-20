from Agent import Agent
from utils import save_training_progress,evaluate
import config
import optuna
import joblib

def optuna_search(trial):
    agent = Agent()
    
    agent.alpha = trial.suggest_float("alpa",0.01,1) # Learning Rate
    agent.gamma = trial.suggest_float("gamma",0.01,1) # Discount Rate
    agent.epsilon = trial.suggest_float("epsilon",0.01,1) # Chance of selecting a random action instead of maximising reward.
    q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain = agent.train()
    if q_table:
        save_training_progress(q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain)
        SR = agent.test()
        print(SR)
        return SR
    else:
        return 0
    

if __name__ == '__main__':
    if config.search_flag:
        study = optuna.create_study()
        study.optimize(optuna_search, n_trials=20, n_jobs=-1)
        joblib.dump(study, config.pickle_name)
    else:
        agent = Agent()

        if config.train_flag:
            q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain = agent.train()
            save_training_progress(q_table,episodes_num_steps,epsiodes_mean_reward,epsiodes_cumulative_reward,episodes_entropy,episodes_penalty,episodes_info_gain)
        else:
            SR = agent.test()
        if config.display_flag:
            agent.display()

        
        # evaluate()
