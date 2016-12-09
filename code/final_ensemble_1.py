
import numpy as np
import pandas as pd

dataY1 = np.loadtxt("../data/final_submission_rfr.csv", skiprows=1, delimiter=",", usecols=(0))
dataY2 = np.loadtxt("../data/final_submission_xgb.csv", skiprows=1, delimiter=",", usecols=(0))

k = 0.5
ensemPred = []
for i, j in zip(dataY1, dataY2):
    ensemPred.append(k*i+(1-k)*j)

df = pd.read_csv('../data/test3.csv')
ids = df['TRIP_ID']
# create submission file
submission = pd.DataFrame(ids, columns=['TRIP_ID'])
submission['TRAVEL_TIME'] = np.exp(ensemPred)
submission.to_csv('../data/final_submission_rfr.csv', index=False)




