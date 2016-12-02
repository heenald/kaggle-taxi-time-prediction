import pandas as pd
import time
import json
import re
import math
import numpy as np
from helper import haversineDistance, CENTER_LATITUDE, CENTER_LONGITUDE

# Convert [lon,lat] string to list
def lonlat_convert(lonlat):
    lon = float(re.compile("[-+]?\d+.\d+").findall(lonlat)[0])
    lat = float(re.compile("[-+]?\d+.\d+").findall(lonlat)[1])
    combined = list()
    combined.append(lon)
    combined.append(lat)
    return combined

def getPoints(polyline):
    print polyline
    points= re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(polyline)
    pointList = list()
    for point in points:
        pointList.append(lonlat_convert(point))
    return pointList

#Gets time in secs
def getTotalTime(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return 15*(len(points)-1)

#Get starting Latitude
def getStartLat(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return float(re.compile("[-+]?\d+.\d+").findall(points[0])[1])


#Get starting Longitude
def getStartLongt(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return float(re.compile("[-+]?\d+.\d+").findall(points[0])[0])

#Get end Latitude
def getEndLat(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return float(re.compile("[-+]?\d+.\d+").findall(points[-1])[1])

#Get end Longitude
def getEndLongt(polyline):
    points = re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(str(polyline))
    return float(re.compile("[-+]?\d+.\d+").findall(points[-1])[0])

#Get haversine distance between starting point and the city center
def getCenterDist(start_lat,start_longt):
    print "here"
    return haversineDistance(np.array([start_lat,start_longt]),np.array([-8.615223,41.157819]))

#Get estimate of direction in which vehicle is moving - ignoring the curvature of earth.
def getDirection(row):
    x1 = row.START_LATITUDE
    x2 = row.END_LATITUDE
    y1 = row.START_LONGITUDE
    y2 = row.END_LONGITUDE
    angle = math.atan2((y2-y1),(x2-x1))*(180/math.pi)
    direction = angle/45
    if direction<0:
        direction +=8
    directions = ['N','NE','E','SE','S','SW','W','NW','N']
    return directions[direction]

#Get bearing (angle of direction) in which vehicle is moving
def getBearing(row):

    lat1=np.radians(row.START_LATITUDE)
    lat2 = np.radians(row.END_LATITUDE)
    longt1 = np.radians(row.START_LONGITUDE)
    longt2 = np.radians(row.END_LONGITUDE)
    dlon = abs(longt2-longt1)
    X = np.cos(lat2)*np.sin(dlon)
    Y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return math.atan2(X,Y)*(180/math.pi)


t0 = time.time()
df = pd.read_csv('../data/train.csv', converters={'POLYLINE': lambda x: json.loads(x)},nrows=5)
df = df[df.MISSING_DATA == False]
#For now removing the examples with empty polyline
df= df[df.POLYLINE!='[]']
print df.shape[0]

df['TOTAL_TIME'] = df.apply(lambda row : getTotalTime(row.POLYLINE), axis=1)
df['START_LATITUDE'] = df.apply(lambda row : getStartLat(row.POLYLINE), axis=1)
df['START_LONGITUDE'] = df.apply(lambda row : getStartLongt(row.POLYLINE), axis=1)
df['END_LATITUDE'] = df.apply(lambda row : getEndLat(row.POLYLINE), axis=1)
df['END_LONGITUDE'] = df.apply(lambda row : getEndLongt(row.POLYLINE), axis=1)
df['DISTANCE_FROM_CENTER'] = df.apply(lambda row : getCenterDist(row['START_LATITUDE'],row['START_LONGITUDE']), axis=1)
df['BEARING'] = df.apply(lambda row : getBearing(row), axis=1)

print time.time()-t0

for i in range(2):
    print df.iloc[[i]]

print df.shape[0]
