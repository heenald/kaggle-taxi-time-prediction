import time
import numpy as np
import pandas as pd
from helper import haversineDistance, CITY_CENTER


def isNear(row1,row2):
    if haversineDistance(np.array([row1['START_LONGT'], row1['START_LAT']]),
                         np.array([row2['START_LONGT'], row2['START_LAT']])) < 0.01:
        if ((row1['BEARING']-row2['BEARING'] >= -50) & (row1['BEARING']-row2['BEARING'] <=50)):
            return True
    return False


def process_row_training(X, row, dftst):
    for i in range(dftest.shape[0]):
        if(isNear(row, dftst.iloc[i])):
            X.append(row)
    return X


t0 = time.time()
df = pd.read_csv('../data/features2.csv')
df_03 = df.sample(frac = 0.3)
dftest = pd.read_csv('../data/test2.csv')

FEATURES_TRAIN = ['WEEK_DAY','HOUR','HOLIDAY','PREV_TO_HOLIDAY','TOTAL_TIME','START_LONGT','START_LAT','START_DIST_FROM_CENTER','BEARING','VELOCITY']
FEATURES_TEST = ['WEEK_DAY','HOUR','HOLIDAY','PREV_TO_HOLIDAY','START_LONGT','START_LAT','START_DIST_FROM_CENTER','BEARING','VELOCITY','TRIP_ID']

print('finding nearest neighbours ...')

X = []
for i in range(df_03.shape[0]):
    if (i%100 == 0):
        print i
        print time.time()
    X = process_row_training(X, df_03.iloc[i], dftest)

ds = pd.DataFrame(X, columns = FEATURES_TRAIN)
ds.columns = FEATURES_TRAIN
ds.to_csv('../data/featuresNN.csv', index=False)

print('Done in %.1f sec.' % (time.time() - t0))