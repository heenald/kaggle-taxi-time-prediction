

import time
import numpy as np
import pandas as pd
import xgboost


t0 = time.time()

X = np.loadtxt("../data/features4Sampled.csv", skiprows=1, delimiter=",", usecols=(0, 1, 5, 6, 7, 8, 9))
y = np.log(np.loadtxt("../data/features4Sampled.csv", skiprows=1, delimiter=",", usecols=(4,)))

# Train
# WEEK_DAY,HOUR,HOLIDAY,PREV_TO_HOLIDAY,TOTAL_TIME,START_LONGT,START_LAT,START_DIST_FROM_CENTER,BEARING,VELOCITY
# 0        1    2       3               4          5           6         7                      8       9

# Test
# WEEK_DAY,HOUR,HOLIDAY,PREV_TO_HOLIDAY,START_LONGT,START_LAT,START_DIST_FROM_CENTER,BEARING,VELOCITY,TRIP_ID
# 0        1    2       3               4           5         6                      7       8         9

X_test = np.loadtxt("../data/test3.csv", skiprows=1, delimiter=",", usecols=(0, 1, 4, 5, 6, 7, 8))
print "Fitting Model"
reg_model = xgboost.XGBRegressor(n_estimators=200)
reg_model.fit(X, y)
print "Predicting"
y_pred = reg_model.predict(X_test)

print time.time()-t0

df = pd.read_csv('../data/test3.csv')
ids = df['TRIP_ID']

# create submission file
submission = pd.DataFrame(ids, columns=['TRIP_ID'])
submission['TRAVEL_TIME'] = np.exp(y_pred)
submission.to_csv('../data/final_submission_gbr.csv', index=False)

print('Done in %.1f sec.' % (time.time() - t0))