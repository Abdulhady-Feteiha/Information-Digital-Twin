import gymnasium as gym
from gym.utils.play import play
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.policies import 
import time


from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
import os
import PIL 


# Create the gym environment
env_id = 'Taxi-v3'
env = gym.make(env_id,render_mode='rgb_array')
#plt.imshow(env.render())
#env = Monitor(env)

# Create a dummy vectorized environment
#env = DummyVecEnv([lambda: env])

# Define the algorithm to use
algorithm = 'DQN'  # Choose from 'PPO', 'A2C', 'DQN'

# Set the hyperparameters
hyperparams = {
    'PPO': {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-3,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'ent_coef': 0.0,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'verbose': 1
    },
    'A2C': {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-3,
        'n_steps': 100,
        'gamma': 0.99,
        'gae_lambda': 1.0,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'rms_prop_eps': 1e-5,
        'use_rms_prop': True,
        'verbose': 1
    },
    'DQN': {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-3,
        'buffer_size': 10000,
        'learning_starts': 1000,
        'batch_size': 64,
        'tau': 0.01,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 1000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.02,
        'verbose': 1
    }
}

# Create the selected algorithm and train the agent
#model = eval(algorithm)(env = env, **hyperparams[algorithm])
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000,progress_bar=True)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render(mode='human')
    #time.sleep(0.2)

# Close the environment
env.close()
