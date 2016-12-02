import numpy as np
import pandas as pd
import re

#Get end Latitude
def getEndLat(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return float(re.compile("[-+]?\d+.\d+").findall(points[-1])[1])

#Get end Longitude
def getEndLongt(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return float(re.compile("[-+]?\d+.\d+").findall(points[-1])[0])

#Returns Haversine distance in KMs - From Kaggle evaluation script
def haversineDistance(p1, p2):
    r = 6371
    p1 = np.array(p1, ndmin=2)
    p2 = np.array(p2, ndmin=2)
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    dlon = abs(p2[:,0] - p1[:,0])
    dlat = abs(p2[:,1] - p1[:,1])
    a = np.sin(dlat)**2 + np.cos(p1[:,1])*np.cos(p2[:,1])*np.sin(dlon)**2
    c = 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c

#Center of city - mean of end points of all trips
CENTER_LATITUDE = -8.615223
CENTER_LONGITUDE = 41.157819

# City center: median of trip end positions
CITY_CENTER    = np.array([[-8.615223, 41.157819]], ndmin=2)

#
# # df = pd.read_csv('../data/train.csv')
# df = df[df.MISSING_DATA == False]
# df= df[df.POLYLINE!='[]']
# print df.shape[0]
# df['END_LATITUDE'] = df.apply(lambda row : getEndLat(row.POLYLINE), axis=1)
# df['END_LONGITUDE'] = df.apply(lambda row : getEndLongt(row.POLYLINE), axis=1)
# lat=0.0
# longt=0.0
# for i in range(df.shape[0]):
#     lat+=df.iloc[[i]].END_LATITUDE
#     longt+=df.iloc[[i]].END_LONGITUDE
#
# lat/=df.shape[0]
# longt/=df.shape[0]
#
# print lat
# print longt