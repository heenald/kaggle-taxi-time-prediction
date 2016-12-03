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
    print len(pln)
    if(len(pln)==1):
        return 0
    for i in range(1, len(pln)):

        temp = haversineDistance(np.array([pln[i - 1][0], pln[i - 1][1]]),
                                 np.array([pln[i][0], pln[i][1]]))
        distanceList.append(temp)
    velocityList = np.array(distanceList)
    medianVelocity = np.median(velocityList)
    return medianVelocity / 15.0

def process_row_training(row):
    pln = row['POLYLINE']
    if (len(pln) < 1):
        return
    pln = np.array(pln, ndmin=2)
    tt = time.localtime(row['TIMESTAMP'])
    data = [tt.tm_wday, tt.tm_hour]
    #Trip started on holiday
    if (row['DAY_TYPE']=="B"):
        data += [1]
    else:
        data += [0]
    #Trip started on a day prior to holiday
    if (row['DAY_TYPE']=="C"):
        data += [1]
    else:
        data += [0]
    data += [(len(pln)-1)*15]
    if (row['CALL_TYPE']=="A"):
        data += [0]
    elif (row['CALL_TYPE']=="B"):
        data += [1]
    else:
        data += [2]
    data += [pln[0][0]]
    data += [pln[0][1]]
    data += [haversineDistance(pln[0, :], CITY_CENTER)[0]]
    data += [getBearing(pln[0][1],pln[0][0],pln[-1][1],pln[-1][0])]
    data += [getVelocity(pln)]
    return pd.Series(np.array(data, dtype=float))

def process_row_test( row):
    pln = row['POLYLINE']
    if(len(pln)<1):
        print "problem!"
    pln = np.array(pln, ndmin=2)
    tt = time.localtime(row['TIMESTAMP'])
    data = [tt.tm_wday, tt.tm_hour]
    if (row['DAY_TYPE']=="B"):
        data += [1]
    else:
        data += [0]
    #Trip started on a day prior to holiday
    if (row['DAY_TYPE']=="C"):
        data += [1]
    else:
        data += [0]
    if (row['CALL_TYPE']=="A"):
        data += [0]
    elif (row['CALL_TYPE']=="B"):
        data += [1]
    else:
        data += [2]
    data += [pln[0][0]]
    data += [pln[0][1]]
    data += [haversineDistance(pln[0, :], CITY_CENTER)[0]]
    data += [getBearing(pln[0][1],pln[0][0],pln[-1][1],pln[-1][0])]
    data += [getVelocity(pln)]
    return pd.Series(np.array(data, dtype=float))


FEATURES_TRAIN = ['WEEK_DAY','HOUR','HOLIDAY','PREV_TO_HOLIDAY','TOTAL_TIME','CALL_TYPE','START_LONGT','START_LAT','START_DIST_FROM_CENTER','BEARING','VELOCITY']
FEATURES_TEST = ['WEEK_DAY','HOUR','HOLIDAY','PREV_TO_HOLIDAY','CALL_TYPE','START_LONGT','START_LAT','START_DIST_FROM_CENTER','BEARING','VELOCITY','TRIP_ID']

t0 = time.time()

print('reading training data ...')
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)})
#
# print df.iloc[[69]]['POLYLINE']
# print getVelocity(df.iloc[[69]]['POLYLINE'])

print('preparing train data ...')
# X = []
# for i in range(df.shape[0]):

#     data = process_row_training(X, df.iloc[i])
#     X.append(data)
#
# print X

ds = df.apply(process_row_training, axis=1)
ds.columns = FEATURES_TRAIN
ds.to_csv('../data/features1.csv', index=False)


print('reading test data ...')
df = pd.read_csv('../data/test.csv', converters={'POLYLINE': lambda x: json.loads(x)})

print df.iloc[[4]]['POLYLINE']
print getVelocity(df.iloc[[5]]['POLYLINE'])


print('preparing test data ...')
dt = df.apply(process_row_test, axis=1)
dt = dt.join(df['TRIP_ID'])
dt.columns = FEATURES_TEST

dt.to_csv('../data/test1.csv', index=False)

print time.time()-t0

