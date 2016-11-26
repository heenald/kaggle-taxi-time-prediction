# kaggle-taxi-time-prediction

Thought Process
1. Call type - likely to have correspondence with trip time. Users prefer a particular way to call for long/short trips.
2. daytype - doesnt differentiate weekday and weekend. can we do that using timestamp, will it help?
3. taxi id - a driver takes particular types of trips?

step1
creating training set with columns:
1.call_type, (categorical)
2.origin call ~ customer id if call type=A, (categorical)
4.taxi id, (categorical)
5.time of day, (range)
6.day type - change A to weekday/weekend, (categorical - 4 types)
7.start lat, (range)
8.start longt, (range)
9.end lat, (range)
10.end longt, (range)
11.start haversine dist from center, (range)
12.direction of travel - divide space into 8 directions, (categorical)
13. total time of travel (range)
ignoring origin stand since it is same as start location.

plot graphs
1. start point, end point
2. start distance from center vs trip time
3. direction vs trip time - 8 graphs (each trip time vs #trips)


Findings:
Total rows in DF - 1710670
Only 10 rows have missing data - can just remove them.

5901 rows have no missing data but empty polyline - i.e. trip shorter than 15 secs. For some start point can be found -
from taxi stand and end point will be very close. But fro rest, can start/end be found at all?
