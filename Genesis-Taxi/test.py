import config
import numpy as np
import pandas as pd
q_table = np.load(config.q_table_DIR)
df = pd.DataFrame(q_table)
df.to_excel("q_table.xlsx")
