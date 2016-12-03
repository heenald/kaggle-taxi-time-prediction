import time
import json
import math
import numpy as np
import pandas as pd
from helper import haversineDistance, CITY_CENTER

#Get bearing (angle of direction) in which vehicle is moving
def getBearing(lat1,longt1,lat2,longt2):

    dlon = abs(longt2-longt1)
    X = np.cos(lat2)*np.sin(dlon)
    Y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return math.atan2(X,Y)*(180/math.pi)

# Get median velocity for each trip
def getVelocity(pln):
    distanceList = list()

    for i in range(1, len(pln)-1):
        temp = haversineDistance(np.array([pln[i - 1][0], pln[i - 1][1]]),
                                 np.array([pln[i][0], pln[i][1]]))
        distanceList.append(temp)

    velocityList = np.array(distanceList)
    medianVelocity = np.median(velocityList)
    return medianVelocity / 15.0

def process_row_training(X, row):
    pln = row['POLYLINE']
    if(len(pln)<1):
        return None
    pln = np.array(pln, ndmin=2)
    tt = time.localtime(row['TIMESTAMP'])
    data = [tt.tm_wday, tt.tm_hour, row['DAYTYPE']]
    data += [(len(pln)-1)*15]
    data += [row['CALL_TYPE']]
    data+=[row['TAXI_ID']]
    data += [pln[0][0]]
    data += [pln[0][1]]
    data += [pln[-1][0]]
    data += [pln[-1][1]]
    data += [haversineDistance(pln[0, :], CITY_CENTER)[0]]
    data += [getBearing(pln[0][1],pln[0][0],pln[-1][1],pln[-1][0])]
    data += [getVelocity(pln)]
    return data

FEATURES = ['wday','hour','daytype','totalTime','call_type','taxi_id','startLongt','startLat','endLongt','endLat','startDistFromCenter','bearing','velocity']
t0 = time.time()

print('reading training data ...')
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})

print('preparing train data ...')
X = []
for i in range(df.shape[0]):
    data = process_row_training(X, df.iloc[i])
    if data != None:
        X.append(data)

print time.time()-t0

