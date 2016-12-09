import time
import numpy as np
import pandas as pd

t0 = time.time()
df = pd.read_csv('../data/features4.csv')
dfs = df.sample(n = 1000000)

dfs.to_csv('../data/features4Sampled_10L.csv', index=False)