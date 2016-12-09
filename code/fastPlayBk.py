import math
import numpy as np
import pandas as pd
from helper import haversineDistance, CITY_CENTER
from random import randint
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

x, y = float(8.61), -1*float(14.23)
lats = []
longs = []
lats.append(x)
longs.append(y)
x, y = float(9.61), -1*float(20.23)
lats.append(x)
longs.append(y)

m = Basemap(llcrnrlon=min(longs),llcrnrlat=min(lats),urcrnrlon=max(longs),urcrnrlat=max(lats),lat_ts=20,
        resolution='c',projection='merc',lon_0=longs[0],lat_0=lats[0])

x1, y1 = m(longs, lats)
m.drawmapboundary(fill_color='white')
m.scatter(x1, y1, s=5, c='r', marker="o")
plt.title("GPS Bus Data")
plt.show()

# df = pd.read_csv('../data/features1.csv', nrows = 100)
# i = randint(0,99)
# testRow = df.iloc[[2]]
# print df.shape[0]
# print df[1:4]
# df =df[:2]+df[3:]
# print df[1:4]
# print testRow
# print df.shape[0]
#
# dataY = np.loadtxt("../data/features2.csv", skiprows=1, delimiter=",", usecols=(4,))
# print np.percentile(np.array(dataY), 10)
# print np.percentile(np.array(dataY), 20)
# print np.percentile(np.array(dataY), 30)
# print np.percentile(np.array(dataY), 40)
# print np.percentile(np.array(dataY), 50)
# print np.percentile(np.array(dataY), 60)
# print np.percentile(np.array(dataY), 70)
# print np.percentile(np.array(dataY), 80)
# print np.percentile(np.array(dataY), 90)
# print np.percentile(np.array(dataY), 100)

def getBearing(lat1,longt1,lat2,longt2):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    longt1 = np.radians(longt1)
    longt2 = np.radians(longt2)
    dlon = longt2-longt1
    X = np.cos(lat2)*np.sin(dlon)
    Y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return np.degrees(math.atan2(X,Y))

def get_bear(lon1, lat1, lon2, lat2):
    lon_diff = (lon1-lon2)*math.pi/180
    b = math.atan2(np.sin(lon_diff)*np.cos(lat2*math.pi/180), np.cos(lat1*math.pi/180)*np.sin(lat2*math.pi/180) - np.sin(lat1*math.pi/180)*np.cos(lat2*math.pi/180)*np.cos(lon_diff))
    return(b/math.pi*180)

def heading(lat1,longt1,lat2,longt2):

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(longt1)
    lon2 = np.radians(longt2)
    aa = np.sin(lon2 - lon1) * np.cos(lat2)
    bb = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(aa, bb) + np.pi

#
# print getBearing(0,0,1,1)
# print get_bear(0,0,1,1)
# print heading(0,0,1,1)
# print
# print getBearing(-10,-10,0,0)
# print get_bear(-10,-10,0,0)
# print heading(-10,-10,0,0)
# print
#
# print getBearing(10,10,0,0)
# print get_bear(10,10,0,0)
# print heading(10,10,0,0)
# print
# print getBearing(-10,-10,0,0)
# print get_bear(-10,-10,0,0)
# print heading(-10,-10,0,0)
# print haversineDistance(np.array([1,0]),np.array([1,1]))

#Get bearing (angle of direction) in which vehicle is moving
# def getBearing(lat1,longt1,lat2,longt2):
#
#     lat1=np.radians(lat1)
#     lat2 = np.radians(lat2)
#     longt1 = np.radians(longt1)
#     longt2 = np.radians(longt2)
#     dlon = abs(longt2-longt1)
#     X = np.cos(lat2)*np.sin(dlon)
#     Y = np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
#     print X
#     print Y
#     return math.atan2(X,Y)*(180/math.pi)
#
# print getBearing(39.099912,-94.581213,38.627089,-90.200203)
#
# arr=np.array([-8.615223,41.157819])
# print arr

for i in range(1, 2):
    print i

print "Hello"