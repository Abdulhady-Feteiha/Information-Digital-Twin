import numpy as np
#arr = np.array([1,0,1,0,1,0])
print(np.load('qq_table.npy')[1794])
#print(np.where((arr == 1))[0])

#print(np.load('./Downloads/Taxi/qq_table.npy'))

#from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete

#observation_space = Discrete(4)
#print(observation_space.sample(np.array([1,1,0,0],dtype=np.int8)))

""" if random.uniform(0, 1) < epsilon:
            next_best_action = env.action_space.sample() # Pick a new action for this state.
            #action = env.action_space.sample()
        else:
            next_best_action = numpy.argmax(q_table[state])# Pick the action which has previously given the highest reward.
        temp_state, temp_reward, temp_done, temp_truncated, temo_info = env.step(next_best_action)
        if temp_state == state:
            temp_reward = -20
            old_temp_value = q_table[next_state, next_best_action] # Retrieve old value from the q-table.
            next_temp_max = numpy.max(q_table[temp_state])
            # Update q-value for next state.
            new_temp_value = (1 - alpha) * old_temp_value + alpha * (temp_reward + gamma * next_temp_max)
            q_table[next_state, next_best_action] = new_temp_value
 """

""" if reward == -20:
        penalties += 2
"""