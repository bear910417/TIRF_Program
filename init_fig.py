import plotly.graph_objects as go
import numpy as np

def init_fig():

    fig = go.Figure()  
   

    fig.add_scatter(x = [], y=[], xaxis='x1', yaxis='y1', name='fret_g', mode='lines', line = {'color': '#013220'}) #0
    fig.add_scatter(x = [], y=[], xaxis='x1', yaxis='y2', name='fret_b', mode='lines', line = {'color': '#00008B'}) #1

    
    fig.add_scatter(x = [], y =[], xaxis='x1', yaxis='y4', name='bb', mode='lines', line = {'color': 'blue'}) #2
    fig.add_scatter(x = [], y =[], xaxis='x1', yaxis='y4', name='bg', mode='lines', line = {'color': 'green'}) #3
    fig.add_scatter(x = [], y =[], xaxis='x1', yaxis='y4', name='br', mode='lines', line = {'color': 'red'}) #4
    fig.add_scatter(x = [], y = [], xaxis='x1',yaxis='y4',name='tot_b', mode='lines', line = {'color': 'black'}) #5


    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y3', name='gg', mode='lines', line = dict(color = 'green'))  #6
    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y3', name='gr', mode='lines', line = dict(color = 'red')) #7
    fig.add_scatter(x = [], y = [],xaxis='x1', yaxis='y3',name='tot_g', mode='lines', line = dict(color = 'black')) #8

    fig.add_scatter(x = [], y = [],xaxis='x1', yaxis='y3',name='rr', mode='lines', line = dict(color = 'orange')) #9



    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y1', name = 'fret_g_bkps', mode = 'markers', marker = dict(color='red', size=10)) #10
    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y2', name='fret_b_bkps', mode='markers', marker = dict(color='red', size=10)) #11
    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y4', name='b_bkps', mode='markers', marker = dict(color='red', size=10)) #12
    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y3', name='g_bkps', mode='markers', marker = dict(color='red', size=10)) #13
    fig.add_scatter(x = [], y = [], xaxis='x1', yaxis='y3', name='r_bkps', mode='markers', marker = dict(color='red', size=10)) #14

    fig.add_histogram(y = [], xaxis='x2',  yaxis='y1', name='Histogram_g', histnorm='probability density', marker = dict(color='#1f77b4', opacity=0.7)) #15
    fig.add_histogram(y = [], xaxis='x2',  yaxis='y2', name='Histogram_b', histnorm='probability density', marker = dict(color='#1f77b4', opacity=0.7)) #16
    #fig.add_trace(go.Image(z = []))


        
    fig.layout = dict(xaxis1 = dict(domain = [0, 0.9]),
                    margin = dict(t = 50),
                    hovermode = 'closest',
                    bargap = 0,
                    uirevision = True,
                    xaxis2 = dict(domain = [0.9, 1]),
                    yaxis1 = dict(domain = [0.01, 0.25]),
                    yaxis2 = dict(domain = [0.26, 0.5]),
                    yaxis3 = dict(domain = [0.51, 0.75]),
                    yaxis4 = dict(domain = [0.76, 1.00]),
                    height = (1000)
                    )

    fig.update_layout(
        xaxis1= dict(
            showline = False,
            showgrid = False,
            showticklabels = True,
            showspikes = True,
            ticks = 'inside',
            range = (0, 360)
        ),
        
        yaxis1 = dict(
            showgrid = True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash='longdash',
            dtick = 0.1,
            showspikes = True,
            autorange = False,
            range = [0, 1]

        ),

        yaxis2 = dict(
            showgrid = True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash='longdash',
            dtick = 0.1,
            showspikes = True,
            autorange = False,
            range = [0, 1]

        ),
        
        yaxis3 = dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash = 'longdash',
            showspikes=True,
            range = (0, None),
        ),
        
        yaxis4=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            showticklabels=True,
            griddash = 'longdash',
            showspikes=True,
            range = (0, None),
        ),
        
        autosize = True,
        showlegend = False,
        xaxis_title='time (s)',
        yaxis_title='FRET'
    )

   
    

    fig2 = go.FigureWidget()
    fig2.add_histogram(x = [] ,name='Dwell_time', histnorm='probability density', marker=dict(color='#f7e1a0', opacity=0.7),
                    xbins=dict(start = 0,end = 1,size=0.02),
                    cumulative=dict(enabled=False))


    fig2.update_layout(
        xaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=True,
            showspikes=True,
            ticks='inside',
            dtick= 0.1,
            range=(0,1)
        ),
        yaxis=dict(rangemode = 'tozero'),
        autosize=True,
        showlegend=False,
        xaxis_title='FRET',
    )
    
    return fig
