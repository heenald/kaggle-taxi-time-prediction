
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


t0 = time.time()

df = pd.read_csv('../data/features2.csv')

# create training set
y = np.log(df['TOTAL_TIME'])
# remove non-predictive features
df.drop(['TOTAL_TIME'], axis=1, inplace=True)
X = np.array(df, dtype=np.float)

print np.any(np.isnan(X))

print('training a random forest regressor ...')
# Initialize the famous Random Forest Regressor from scikit-learn
clf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=21)
clf.fit(X, y)

print('predicting test data ...')
df = pd.read_csv('../data/test2.csv')
ids = df['TRIP_ID']

df = df.drop(['TRIP_ID'], axis=1)
X_tst = np.array(df, dtype=np.float)
print np.any(np.isnan(X))
y_pred = clf.predict(X_tst)

# create submission file
submission = pd.DataFrame(ids, columns=['TRIP_ID'])
submission['TRAVEL_TIME'] = np.exp(y_pred)
submission.to_csv('../data/submission2', index=False)

print('Done in %.1f sec.' % (time.time() - t0))