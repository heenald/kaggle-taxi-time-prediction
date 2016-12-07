import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import ShuffleSplit, KFold

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

df = pd.read_csv('../data/features3WithoutEmptyLines.csv', nrows=40)
df=df[20:40]
print df.info()


df=df[df.TOTAL_TIME > 0]

y = np.log(df['TOTAL_TIME'])
df.drop(['TOTAL_TIME','HOLIDAY','PREV_TO_HOLIDAY'], axis=1, inplace=True)
X = np.array(df, dtype=np.float)

print np.isnan(X.any())
print np.isfinite(X.all())
print np.isnan(y.any())
print np.isfinite(y.all())

print('Training set size: %i x %i' % X.shape)

y_pred_rf = np.zeros(y.shape)
for trn_idx, val_idx in KFold(X.shape[0], n_folds=2):
    # split training data
    X_trn, X_tst, y_trn, y_tst = X[trn_idx, :], X[val_idx, :], y[trn_idx], y[val_idx]
    print y_tst.shape

    # Initialize the famous Random Forest Regressor from scikit-learn
    clf = RandomForestRegressor(n_jobs=4)
    print X_trn[0]
    clf.fit(X_trn, y_trn)
    y_pred_rf[val_idx] = clf.predict(X_tst)
    print y_pred_rf[0]

    # or the Gradient Boosting Regressor
    # clf = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=23)
    # clf.fit(X_trn, y_trn)
    # y_pred_gb[val_idx] = clf.predict(X_tst)

    print('  Score RFR: %.4f' % (rmse(y_tst, y_pred_rf[val_idx])))#, rmse(y_tst, y_pred_gb[val_idx])))

# save prediction result to file
err_rf = rmse(y, y_pred_rf)
#err_gb = rmse(y, y_pred_gb)
print {'size': X.shape[0], 'rf': err_rf}#, 'gb': err_gb}
