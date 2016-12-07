import time
import numpy as np
import pandas as pd

t0 = time.time()
df = pd.read_csv('../data/features3.csv')
df_02 = df.sample(frac = 0.2)

df_02.to_csv('../data/features3Sampled.csv', index=False)