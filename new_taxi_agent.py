from new_env import TaxiEnvCustomized
import gym
from gym.core import RenderFrame
import numpy
import random
from os import system, name
from time import sleep

# Define function to clear console window.
def clear(): 
  
    # Clear on Windows.
    if name == 'nt': 
        _ = system('cls')
  
    # Clear on Mac and Linux. (os.name is 'posix') 
    else: 
        _ = system('clear')

clear()

"""Setup"""
#1463 1663 992
#env = TaxiEnvCustomized(render_mode='rgb_array')
env = TaxiEnvCustomized(render_mode='human')
#env = TaxiEnvCustomized(render_mode='ansi')


# Make a new matrix filled with zeros.
# The matrix will be 2000x6 as there are 2000 states and 6 actions.
#q_table = numpy.ones([env.observation_space.n, env.action_space.n])*(-50)
#q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

q_table = numpy.load("qq_table.npy")

training_episodes = 40000 # Amount of times to run environment while training.
display_episodes = 10 # Amount of times to run environment after training.

# Hyperparameters
alpha = 0.1 # Learning Rate
gamma = 0.6 # Discount Rate
epsilon = 0.5 # Chance of selecting a random action instead of maximising reward.

# For plotting metrics
all_epochs = []
all_penalties = []

"""Training the Agent"""

""" for i in range(training_episodes):
    resetter = env.reset()  #Reset returns observation state and other info.
    state = resetter[0]
    info = resetter[1]
    done = False
    penalties, reward, = 0, 0
    #print(info["action_mask"].dtype)

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Pick a new action for this state.
            #action = env.action_space.sample()
        else:
            action = numpy.argmax(q_table[state])# Pick the action which has previously given the highest reward.
            #action = numpy.argmax(q_table[state])
            #action = numpy.argmax(q_table[state, numpy.where(info["action_mask"] == 1)[0]])

        next_state, reward, done, truncated, info = env.step(action)
       
        
        
        old_value = q_table[state, action] # Retrieve old value from the q-table.
        next_max = numpy.max(q_table[next_state])

        # Update q-value for current state.
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10: # Checks if agent attempted to do an illegal action.
            penalties += 1
        

        state = next_state
        
    if i % 100 == 0: # Output number of completed episodes every 100 episodes.
        print(f"Episode: {i}")

print("Training finished.\n")
numpy.save("qq_table.npy",q_table) """


"""Display and evaluate agent's performance after Q-learning."""

total_epochs, total_penalties = 0, 0

for _ in range(display_episodes):
    state,info= env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = numpy.argmax(q_table[state])
        state, reward, done, truncated,info = env.step(action)

        if reward == -10:
            penalties += 1
        
        epochs += 1
        clear()
        env.render()
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        sleep(0.15) # Sleep so the user can see the 

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {display_episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / display_episodes}")
print(f"Average penalties per episode: {total_penalties / display_episodes}")