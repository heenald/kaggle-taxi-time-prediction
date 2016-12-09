import plotly
print plotly.__version__  # version >1.9.4 required
from plotly.graph_objs import Scatter, Layout
import pandas as pd


df = pd.read_csv('../data/startEndAll.csv', nrows = 5)
df.head(2)

plotly.offline.plot({
    'data': [
  		{
  			'x': df.START_LONGT,
        	'y': df.START_LAT,
        	'mode': 'markers',
        	'name': 'start'},
        {
        	'x': df.END_LONGT,
        	'y': df.END_LAT,
        	'mode': 'markers',
        	'name': 'end'}
    ],
    'layout': {
        'xaxis': {'title': 'Longitude'},
        'yaxis': {'title': "Latitude"}
    }
})