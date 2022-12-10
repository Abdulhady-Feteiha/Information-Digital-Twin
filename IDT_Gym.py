class IDT_Gym:

    def __init__(self):
        """
        Initialize The IDT_Gyme environment
        """      
        pass
    def load_IHM_Dataset(self):
        """
        Load The IHM dataset from excel into a pandas DataFrame
        """
        pass
    def Episodise(self):
        """
        Find identical states in a dataset, create a new dataset with all applicable actions/next states
        """
        pass

    def fetsh_record(self):
        """
        Giiven the passesd state-action pair, retrieve: next state, all applicable actions
        """
        pass

    def execute_action(self):
        """
        Update the IHM with the executed action
        """
        pass

    def get_reward(self):
        """
        Given the passed state-action pair, retrieve: the reward from the IHM
        """
        pass

    def set_reward(self):
        """
        Given the passed state-action pair, retrieve: the reward from the IHM
        """
        pass
    def reset(self):
        '''
        Resets variables necessary to start next training episode.
        '''
if __name__ == '__main__':
    ENV = IDT_Gym()

