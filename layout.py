from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import dash_daq as daq

def make_app():
    app = Dash(__name__, external_stylesheets = [dbc.themes.LUMEN], prevent_initial_callbacks = True)
    events = [{'event': 'keydown', 'props': ['key']}]   
    app.layout = html.Div([
        html.H2('Trace Viewer',style={'textAlign': 'center', 'padding': 5}),
        EventListener(id = 'key_events', events = events),    
        dcc.Graph(id="graph", config = {'doubleClick' : 'false'}),     
        dcc.Tabs(id='tabs', value='Tools', children=[
        dcc.Tab(label = 'Tools', value = 'Tools', children=[          
            html.Div([
                html.Div('Hide Trace:', style={"margin-left": "60px",'padding': 5 }),
                html.Div([dcc.Checklist(
                        ['BB', 'BG', 'BR', 'GG', 'GR', 'RR', 'FRET BG', 'FRET GR', 'Tot B', 'Tot G'],
                        ['Tot B', 'Tot G'],
                        inline = True,
                        inputStyle={"margin-left": "10px"},
                        id='show',
                        persistence = True
                        )],
                        style={'padding': 5}),  
                html.Div('Smoothing',style={"margin-left": "10px"}),
                dcc.Input(1, type = 'number', min = 1, step = 1, id = 'smooth', persistence = 'True', style={"margin-left": "10px", 'width': '50px'}), 
                html.Div('Scatter:', style={"margin-left": "20px",'padding': 5 }),
                daq.ToggleSwitch(id = 'scatter', value = 0, color = 'green', style={"margin-left": "10px",'padding': 5}), 

                html.Div([          
                        dcc.RadioItems(
                            ['Add', 'Remove','Except',"Clear","Clear All", "Set All"],
                            'Add',
                            id='AR',
                            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                        )
                    ],style={'padding': 5, "margin-left": '40%'})
                    
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items' : 'center'}),
        
            
            html.Div([html.Button('Set Dead time', id='dtime'),
                    html.Button('Set End time', id='etime'),
                    html.Button('Previous', id='previous',accessKey = 'q'),
                    html.Button('Next', id='next', accessKey = 'w'), 
                    html.Button('Good', id='set_good'),
                    html.Button('Bad', id='set_bad'),
                    html.Button('Save Selected', id='select')],
                    style={"margin-left": "60px",'padding': 5, 'flex': 1.5}),   

            html.Div([dcc.Input(id = "bkp", type="text", placeholder="", style={'textAlign': 'center'},size='20'),
                    dcc.Input(id = "b_bkp", type="text", placeholder="", style={'textAlign': 'center'},size='20'),
                    dcc.Input(value = 0, id = "i", type="text", placeholder="", style={'textAlign': 'center'}, size='3', persistence = 'True'),
                    html.Button('Go', id='tr_go'),
                    dcc.Loading(id="loading1",type="default", children = html.Div('Total_traces: '+ str(0), id='N_traces',style={"margin-left": "10px"}))],
                    style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),
            
            
            html.Div([html.Button('Save breakpoints', id='save_bkps'),
                        html.Button('Load breakpoints', id='load_bkps'),
                        html.Button('Rupture', id='rupture', disabled = True),
                        html.Button('Rescale', id='rescale'),
                        dcc.Dropdown([], 'fret_g', clearable = False, searchable = False, style = {'width' : '100px'}, id='channel'),            
            ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),
            
            html.Div([
                html.Div('Path:'),
                dcc.Input(id="path", type="text", placeholder="", style={'textAlign': 'left'}, size='50', persistence = 'True'),
                html.Button('Load', id='loadp')
                ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),
        ], style = {'width': '95%', 'height' : '20%', 'padding' : 5}),




        dcc.Tab(label='Aois', value = 'Aois', children=[  
            html.Div([ 

            dcc.Graph(id = "g_blob", config={'displayModeBar': False, 'staticPlot' : True}) 

            ],style={'padding': 5, 'display': 'flex', 'flex-direction': 'row', 'align-items' : 'left'})

        ], style = {'width': '90%', 'height' : '20%', 'padding' : 5}),
        ], vertical = True, style = {'width' : "5%", "heigth" : '20%'}, content_style = {'width' : "95%", "heigth" : '20%'}, parent_style ={'width' : "100%", "heigth" : '20%'}),    






        ##############
        html.Div([
            dcc.Loading(id="loading",type="default", children =  dcc.Graph(id="Hist")),

            ],style={'padding': 5,'width' : "60%",'float': 'left', 'display': 'inline-block'}
            ),
        html.Div([
            
            html.Div('Fit n Gaussian Peaks: '),
            dcc.RadioItems(
                ['0', '1', '2', '3', '4', '5', '6', '7', '8'],
                '2',
                id = 'fmode',
                labelStyle={'display': 'inline-block', 'marginTop': '5px', "margin-left": "20px"}
            ),

            
            html.Br(),
            html.Div('Binsize:'),      
            dcc.Slider(
                0.01,
                0.1,
                step=0.01,
                id='binsize',
                value=0.02,
            ),
            html.Button('Fit Histogram', id='fitd'),
            html.Button('Save Histogram', id='save_d'),
            ], style={'margin-top':'100px','width': '28%', 'float': 'middle', 'display': 'inline-block'})
            
            

    ])
    
    return app