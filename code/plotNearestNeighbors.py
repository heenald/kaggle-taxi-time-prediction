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

t0 = time.time()

df = pd.read_csv('../data/startEndAll.csv', nrows = 500000)
i = 34241 #randrange(df.shape[0])
#print i
testRow = df.iloc[[i]]
#print testRow
df = df.drop(i)

X = pd.DataFrame()
for i in range(df.shape[0]):
    if i%1000 ==0:
        print i
        # print X.shape[0]
    if (isNear(testRow, df.iloc[[i]])):
        X=X.append(df.iloc[[i]])

frequencies= collections.OrderedDict(sorted(X['TOTAL_TIME'].value_counts().to_dict().items()))
print frequencies
print testRow['TOTAL_TIME']

plotly.offline.plot({
    'data': [
  		{
  			'x': X.START_LONGT,
        	'y': X.START_LAT,
        	'mode': 'markers',
        	'name': 'start'},
        {
        	'x': X.END_LONGT,
        	'y': X.END_LAT,
        	'mode': 'markers',
        	'name': 'end'},
        {
  			'x': testRow.START_LONGT,
        	'y': testRow.START_LAT,
        	'mode': 'markers',
        	'name': 'start'},
        {
        	'x': testRow.END_LONGT,
        	'y': testRow.END_LAT,
        	'mode': 'markers',
        	'name': 'end'}
    ],
    'layout': {
        'xaxis': {'title': 'Longitude'},
        'yaxis': {'title': "Latitude"}
    }
})


fig, ax = plt.subplots()

colors = list("rgbcmyk")
x = frequencies.keys()
y = frequencies.values()
plt.bar(x,y,color=colors.pop())

fig.savefig('../data/fig.jpeg')


figAll, ax = plt.subplots()
frequenciesAll = df['TOTAL_TIME'].value_counts().to_dict()
v=list(frequenciesAll.values())
k=list(frequenciesAll.keys())
print k[v.index(max(v))]
frequenciesAll= collections.OrderedDict(sorted(frequenciesAll.items()))
x = frequenciesAll.keys()
y = frequenciesAll.values()
plt.bar(x,y,color=colors.pop())

figAll.savefig('../data/figAll.jpeg')
print time.time()-t0






