import time
import json
import math
import collections
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from helper import haversineDistance, CITY_CENTER
from random import randrange
import plotly.plotly as py
from plotly.graph_objs import *
import plotly
#print plotly.__version__  # version >1.9.4 required
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objs as go


def isNear(row1, row2):

    start1Longt = row1.iloc[0]['START_LONGT']
    start1Lat = row1.iloc[0]['START_LAT']
    start2Longt = row2.iloc[0]['START_LONGT']
    start2Lat = row2.iloc[0]['START_LAT']
    hsDistanceBetweenStarts = haversineDistance(np.array([start1Longt, start1Lat]),
                                 np.array([start2Longt, start2Lat]))
    if ((hsDistanceBetweenStarts< -0.25) or (hsDistanceBetweenStarts>0.25)):
        return False
    if ((row1.iloc[0]['BEARING']-row2.iloc[0]['BEARING']<-30) or (row1.iloc[0]['BEARING']-row2.iloc[0]['BEARING']>30)):
        return False

    return True



df = pd.read_csv('../data/features4Sampled.csv')

testdf = pd.read_csv('../data/test3.csv')

for t in range(200,220):
    t0 = time.time()
    print "Test example:",t

    testRow = testdf.iloc[[t]]

    X = pd.DataFrame()
    for i in range(df.shape[0]):
        if i%10000 ==0:
            print "Progressed to:",i
        if (isNear(testRow, df.iloc[[i]])):
            X=X.append(df.iloc[[i]])

    X.to_csv('../data/train_pp_TST_%i.csv' % (t),index=False)

    print "Finished processing %i in " % (t), time.time()-t0






