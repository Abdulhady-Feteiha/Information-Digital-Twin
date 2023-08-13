import config
import numpy as np
import pandas as pd
q_table = np.load("Q_tables/original_env.npy")
df = pd.DataFrame(q_table)
df.to_excel("q_table.xlsx")
