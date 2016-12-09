import numpy as np
import os
import time
import numpy as np
import cPickle as pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost
from matplotlib import plot as plt

df = pd.read_csv('../data/features4Sampled.csv')
print df.info()
X = np.array(df, dtype=np.float)

Y = np.log(df['TOTAL_TIME']+1)
df.drop(['TOTAL_TIME','HOLIDAY','PREV_TO_HOLIDAY'], axis=1, inplace=True)


Xtr = X[0:400000]
Ytr = Y[0:400000]
Xtest = X[400001:]
Ytest = Y[400001:]

rfr_model = RandomForestRegressor(n_estimators=200)
rfr_model.fit(Xtr, Ytr)
y_rfr = rfr_model.predict(Xtr)

xgb_model = xgboost.XGBRegressor(n_estimators=200)
xgb_model.fit(Xtr, Ytr)
y_xgb = xgb_model.predict(Xtr)

wt = [0.6,0.65,0.7,0.75, 0.8]

r2_score=[]

for i,k in enumerate(wt):
    Yfinal = k * y_rfr + (1-k) * y_xgb;
    temp = r2_score(Ytest, Yfinal)
    r2_score.append(temp)



print r2_score
plt.plot(wt,auc,marker='o');
plt.show()



