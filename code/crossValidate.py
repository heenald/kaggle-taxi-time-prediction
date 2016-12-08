import time
import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    # type: (object, object) -> object
    return np.sqrt(np.mean((y_true - y_pred)**2))

df = pd.read_csv('../data/features4Sampled.csv')
print df.info()

y = np.log(df['TOTAL_TIME']+1)
df.drop(['TOTAL_TIME','HOLIDAY','PREV_TO_HOLIDAY'], axis=1, inplace=True)
X = np.array(df, dtype=np.float)

print np.isnan(X.any())
print np.isfinite(X.all())
print np.isnan(y.any())
print np.isfinite(y.all())

print('Training set size: %i x %i' % X.shape)

y_pred_rf = np.zeros(y.shape)
y_pred_gb = np.zeros(y.shape)


dataX = np.loadtxt("../data/features4Sampled.csv", skiprows=1, delimiter=",", usecols=(0, 1, 5, 6, 7, 8, 9))
dataY = np.log(np.loadtxt("../data/features4Sampled.csv", skiprows=1, delimiter=",", usecols=(4,)))

# Train
# WEEK_DAY,HOUR,HOLIDAY,PREV_TO_HOLIDAY,TOTAL_TIME,START_LONGT,START_LAT,START_DIST_FROM_CENTER,BEARING,VELOCITY
# 0        1    2       3               4          5           6         7                      8       9

# Test
# WEEK_DAY,HOUR,HOLIDAY,PREV_TO_HOLIDAY,START_LONGT,START_LAT,START_DIST_FROM_CENTER,BEARING,VELOCITY,TRIP_ID
# 0        1    2       3               4           5         6                      7       8         9


X = dataX
y = dataY

for train_idx, test_idx in KFold(X.shape[0], n_folds=5):
    t0 = time.time()
    X_train, X_test, y_train, y_test = X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]

    reg_model = RandomForestRegressor(n_estimators=200)
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    # reg_model = RandomForestRegressor()
    # reg_model.fit(X_train,y_train)
    # y_pred_1 = reg_model.predict(X_test)

    #accuracy1 = rmse(y_test, y_pred_1)
    accuracy = rmse(y_test, y_pred)
    print(accuracy)
    #print(accuracy1)
    print time.time()-t0

# for trn_idx, val_idx in KFold(X.shape[0], n_folds=2):
#     # split training data
#     X_trn, X_tst, y_trn, y_tst = X[trn_idx, :], X[val_idx, :], y[trn_idx], y[val_idx]
#     print y_tst.shape
#
#     # Initialize the famous Random Forest Regressor from scikit-learn
#     # clf = RandomForestRegressor(n_jobs=4)
#     # print X_trn[0]
#     # clf.fit(X_trn, y_trn)
#     # y_pred_rf[val_idx] = clf.predict(X_tst)
#     # print y_pred_rf[0]
#
#     # or the Gradient Boosting Regressor
#     # clf = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=23)
#     # clf.fit(X_trn, y_trn)
#     # y_pred_gb[val_idx] = clf.predict(X_tst)
#
#     reg_model = xgboost.XGBRegressor(n_estimators=200)
#     reg_model.fit(X_trn, y_trn)
#     y_pred_gb = reg_model.predict(X_tst)
#     accuracy = rmse(y_tst, y_pred_gb)
#     print(accuracy)
#     print('  Score RFR: %.4f' % (rmse(y_tst, y_pred_gb[val_idx])))#, rmse(y_tst, y_pred_gb[val_idx])))
#
# # save prediction result to file
# err_rf = rmse(y, y_pred_gb)
# #err_gb = rmse(y, y_pred_gb)
# print {'size': X.shape[0], 'rf': err_rf}#, 'gb': err_gb}
