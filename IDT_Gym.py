import openpyxl
from pathlib import Path
import numpy as np 

class IDT_Gym:

    def __init__(self):
        """
        Initialize The IDT_Gyme environment
        """      
        self.load_IHM_Dataset()
        print(self.Name)

    def load_IHM_Dataset(self):
        """
        Load The IHM dataset from excel into a pandas DataFrame
        """
        xlsx_file = Path('IHM', 'Events Matrix_V1.xlsx')
        wb_obj = openpyxl.load_workbook(xlsx_file)
        # print(wb_obj.sheetnames)
        sheet = wb_obj["Episodes"]
        # print(sheet["H33"].value)
        self.Number = np.asarray([])
        self.Category = np.asarray([])
        self.Group = np.asarray([])
        self.UID = np.asarray([])
        self.Name = np.asarray([])
        self.ID = np.asarray([])
        self.Cycle = np.asarray([])
        self.Timee_stamp = np.asarray([])
        self.Inputs = np.zeros((6,60))
        self.Contexts = np.zeros((3,60))
        self.Actions = np.zeros((7,60))
        self.Outputs = np.zeros((9,60))
        # self.States = [Inputs,Context,Output]
        for row in sheet.iter_rows(min_row=18, min_col=8, max_row=18, max_col=32):
            for cell in row:
                self.Number= np.append(self.Number,cell.value) 
        for row in sheet.iter_rows(min_row=19, min_col=8, max_row=19, max_col=32):
            for cell in row:
                self.Category= np.append(self.Category,cell.value) 
        for row in sheet.iter_rows(min_row=20, min_col=8, max_row=20, max_col=32):
            for cell in row:
                self.Group= np.append(self.Group,cell.value) 
        for row in sheet.iter_rows(min_row=21, min_col=8, max_row=21, max_col=32):
            for cell in row:
                self.UID= np.append(self.UID,cell.value) 
        for row in sheet.iter_rows(min_row=22, min_col=8, max_row=22, max_col=32):
            for cell in row:
                self.Name= np.append(self.Name,cell.value) 
        for row in sheet.iter_rows(min_row=31, min_col=4, max_row=90, max_col=4):
            for cell in row:
                self.ID= np.append(self.ID,cell.value) 
        for row in sheet.iter_rows(min_row=31, min_col=5, max_row=90, max_col=5):
            for cell in row:
                self.Cycle= np.append(self.Cycle,cell.value) 
        for row in sheet.iter_rows(min_row=31, min_col=6, max_row=90, max_col=6):
            for cell in row:
                self.Timee_stamp= np.append(self.Timee_stamp,cell.value) 
        i = 0
        for row in sheet.iter_rows(min_row=31, min_col=8, max_row=90, max_col=13):
            for j,cell in enumerate(row):
                if cell.value:
                    self.Inputs[i,j] = cell.value
            i+=1
        i = 0
        for row in sheet.iter_rows(min_row=31, min_col=8, max_row=90, max_col=13):
            for j,cell in enumerate(row):
                if cell.value:
                    self.Contexts[i,j] = cell.value
        for row in sheet.iter_rows(min_row=31, min_col=8, max_row=90, max_col=32):
            for j,cell in enumerate(row):
                self.Outputs[i,j] = cell.value
        for row in sheet.iter_rows(min_row=31, min_col=8, max_row=90, max_col=32):
            for j,cell in enumerate(row):
                self.Actions[i,j] = cell.value
        pass
    def Episodise(self):
        """
        Find identical states in a dataset, create a new dataset with all applicable self.Actionss/next states
        """
        pass

    def fetsh_record(self):
        """
        Giiven the passesd state-self.Actions pair, retrieve: next state, all applicable self.Actionss
        """
        pass

    def execute_action(self):
        """
        Update the IHM with the executed self.Actions
        """
        pass

    def get_reward(self):
        """
        Given the passed state-self.Actions pair, retrieve: the reward from the IHM
        """
        pass

    def set_reward(self):
        """
        Given the passed state-self.Actions pair, retrieve: the reward from the IHM
        """
        pass
    def reset(self):
        '''
        Resets variables necessary to start next training episode.
        '''
if __name__ == '__main__':
    ENV = IDT_Gym()

