import openpyxl
from pathlib import Path
import numpy as np 
import pandas as pd

class Genesis_RL:

    def __init__(self):
        """
        Initialize The RL algorithm
        """      
        ENV = IDT_Gym()
        ENV.Episodise()
        ENV.load_IHM_Dataset()
        
    def load_Q_table(self):
        """
        Initialize a Q table if it is the first cycle, or load saved Q table instead
        """
        pass
    def Takte_action(self):
        """
        Given a state, available actions pair, return the choosen action
        """
        pass
    def Get_reward(self):
        """
        Giiven the passesd state-Actions pair, retrieve: rewrd 
        """ 
  
    def Calculate_bellman_eq(self):
        """
        Given current reward,and loaded Q table. Retrieve current Q value
        """
        pass

    def Update_Q_table(self):
        """
        Update the Q table with current values
        """
        pass

    def Recommend_action(self):
        """
        Given the Q table and current sate, chose an action
        """
        pass

    def save_Q_table(self):
        """
        Save updated Q table
        """
        pass


    def reset(self):
        '''
        Resets variables necessary to start next training episode.
        '''
if __name__ == '__main__':

    Model = Genesis_RL()
    # next_s,available_actions = (ENV.fetsh_record("S3","A1"))
    # for i in range(len(ENV.Inputs)):
    #     print(ENV.Inputs[i])

