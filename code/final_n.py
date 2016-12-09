import os
import time
import numpy as np
import pandas as pd
import xgboost

y_finalPred = []

#WEEK_DAY,HOUR,HOLIDAY,PREV_TO_HOLIDAY,TOTAL_TIME,START_LONGT,START_LAT,START_DIST_FROM_CENTER,BEARING,VELOCITY
#0          1   2       3               4           5           6       7                       8

for i in range(1,41):
    t0 = time.time()
    print i
    X = np.loadtxt('../data/train_pp_TST_%i.csv' % (i), skiprows=1, delimiter=",", usecols=(0, 1, 5, 6, 7, 8))
    y = np.log(np.loadtxt("../data/train_pp_TST_%i.csv" % (i), skiprows=1, delimiter=",", usecols=(4,)))
    if(y.size < 682):
        y_finalPred.append(-1)
        break
    X_test = np.loadtxt("../data/test3.csv", skiprows=1, delimiter=",", usecols=(0, 1, 4, 5, 6, 7, 8))
    reg_model = clf = xgboost.XGBRegressor(n_estimators=200)
    reg_model.fit(X, y)
    y_pred = reg_model.predict(X_test)
    y_finalPred.append(y_pred)
    # create submission file
    print('Done in %.1f sec.' % (time.time() - t0))

df = pd.read_csv('../data/test3.csv')
ids = df['TRIP_ID']

submission = pd.DataFrame(ids, columns=['TRIP_ID'])
submission['TRAVEL_TIME'] = np.exp(y_finalPred)
submission.to_csv('../data/final_submission_n_%i.csv' %i, index=False)
print('Done in %.1f sec.' % (time.time() - t0))
print "Finished processing %i in " % (i), time.time()-t0