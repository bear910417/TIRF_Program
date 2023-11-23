from dash_extensions.enrich import DashProxy, dcc, html, Output, Input, BlockingCallbackTransform, callback_context,  dash_table
from dash_extensions import EventListener
import dash_bootstrap_components as dbc
import dash_daq as daq



def make_app(fig, fig_blob, fig2):
    app = DashProxy(__name__, external_stylesheets = [dbc.themes.LUMEN], prevent_initial_callbacks = True, transforms=[BlockingCallbackTransform(timeout = 1)])
    events = [{'event': 'keydown', 'props': ['key']}]   
    app.layout = html.Div([
        EventListener(id = 'key_events', events = events),    
        dcc.Graph(id="graph", figure = fig, config = {'doubleClick' : False}),     
        dcc.Tabs(id='tabs', value='Tools', children=[
        #TOOLS#
        dcc.Tab(label = 'Tools', value = 'Tools', children=[          
            html.Div([
                html.Div('Hide Trace:', style={"margin-left": "60px",'padding': 5 }),
                html.Div([dcc.Checklist(
                        ['BB', 'BG', 'BR', 'GG', 'GR', 'RR', 'FRET BG', 'FRET GR', 'Tot B', 'Tot G', 'HMM'],
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
                html.Div('Strided Smooth:', style={"margin-left": "20px",'padding': 5 }),
                daq.ToggleSwitch(id = 'strided', value = 0, color = 'green', style={"margin-left": "10px",'padding': 5}),

                html.Div([          
                        dcc.RadioItems(
                            ['Add', 'Remove','Except',"Clear","Clear All", "Set All", "Reset"],
                            'Add',
                            id='AR',
                            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                        )
                    ],style={'padding': 5, "margin-left": '30%'}),
                dcc.ConfirmDialog(
                    id = 'confirm-reset',
                    message = 'All breakpoint will be deleted, are you sure you want to continue?',
                ),
                    
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
                    dcc.Input(value = 0, id = "i", type = 'number', placeholder="", style={'textAlign': 'center', 'width': '80px'}, persistence = 'True'),
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

            html.Div([
                dbc.InputGroup([
                    dbc.InputGroupText("Find the"),
                    dcc.Dropdown(['first', 'second', 'previous'], 'previous', id = 'chp_mode_0', clearable = False, searchable = False, style={'textAlign': 'center', 'width': '100px'}, persistence = 'True'),
                    dbc.InputGroupText("value that is "),
                    dcc.Dropdown(['bigger', 'smaller'], 'smaller', id = 'chp_comp_0', clearable = False, searchable = False, style={'textAlign': 'center', 'width': '100px'}, persistence = 'True'),
                    dbc.InputGroupText("than "),
                    dbc.Input(value = 0.5, type="number", id = 'chp_thres_0', size = 5, step = 0.05, style={'textAlign': 'center', 'width': '80px'}, persistence = 'True'),
                    dbc.InputGroupText("in "),
                    dcc.Dropdown([], 'fret_g', id='chp_channel_0', clearable = False, searchable = False, style = {'width' : '100px'}, persistence = 'True', persisted_props = ['value', 'options']),   
                    dbc.InputGroupText("for "),
                    dcc.Dropdown(['current trace', 'all traces', 'all good'], 'current trace', id = 'chp_target_0', clearable = False, searchable = False, style={'textAlign': 'center', 'width': '150px'}, persistence = 'True'),
                    dbc.Button("Find", id="chp_find_0", n_clicks=0),
                ], size = 'small')

            ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row', 'width': '40%'}),

            html.Div([
                dbc.InputGroup([
                    dbc.InputGroupText("Find the"),
                    dcc.Dropdown(['first', 'second', 'previous'], 'previous', id = 'chp_mode_1', clearable = False, searchable = False, style={'textAlign': 'center', 'width': '100px'}, persistence = 'True'),
                    dbc.InputGroupText("value that is "),
                    dcc.Dropdown(['bigger', 'smaller'], 'smaller', id = 'chp_comp_1', clearable = False, searchable = False, style={'textAlign': 'center', 'width': '100px'}, persistence = 'True'),
                    dbc.InputGroupText("than "),
                    dbc.Input(value = 0.5, type="number", id = 'chp_thres_1', size = 5, step = 0.05, style={'textAlign': 'center', 'width': '80px'}, persistence = 'True', persisted_props = ['value', 'options']),
                    dbc.InputGroupText("in "),
                    dcc.Dropdown([], 'fret_b', id='chp_channel_1', clearable = False, searchable = False, style = {'width' : '100px'}, persistence = 'True'),   
                    dbc.InputGroupText("for "),
                    dcc.Dropdown(['current trace', 'all traces', 'all good'], 'current trace', id = 'chp_target_1', clearable = False, searchable = False, style={'textAlign': 'center', 'width': '150px'}, persistence = 'True'),
                    dbc.Button("Find", id="chp_find_1", n_clicks=0),
                ], size = 'small')

            ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row', 'width': '40%'})
            
        ], style = {'width': '95%', 'height' : '20%', 'padding' : 5}),



        #AOIS#
        dcc.Tab(label='Aois', value = 'Aois', children=[  
            html.Div([ 

            dcc.Graph(id = "g_blob", figure = fig_blob, config={'displayModeBar': True, 'staticPlot' : False}) 

            ],style={'padding': 5, 'display': 'flex', 'flex-direction': 'row', 'align-items' : 'left'}),

            html.Div([ 
                html.Div('max: '),
                dcc.Slider(0, 10000, 100, value = 0, updatemode='drag',
                tooltip={"placement": "bottom", "always_visible": False}, marks = None, id = 'aoi_max')

            ],style={'padding': 5, 'width': '90%'}),

        ], style = {'width': '90%', 'height' : '20%', 'padding' : 5}),


        #HMM#
        dcc.Tab(label='HMM', value = 'HMM', children=[  
            html.Div([
                html.Div('Fit:'),
                daq.ToggleSwitch(id = 'hmm_fit', value = True, color = 'green', style={"margin-left": "10px"}),
                html.Div('Fix means:', style={"margin-left": "10px"}),
                daq.ToggleSwitch(id = 'hmm_fix', value = 0, color = 'green', style={"margin-left": "10px"}),
                html.Div('Plot:', style={"margin-left": "10px"}),
                daq.ToggleSwitch(id = 'hmm_plot', value = 0, color = 'green', style={"margin-left": "10px"}),
            ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),

            html.Div([
                html.Div('Cov Type: '),
                dcc.RadioItems(
                            ['spherical', 'diag', 'full',"tied"],
                            'spherical',
                            id='hmm_cov_type',
                            labelStyle={'display': 'inline-block', "margin-left": "10px"}
                        ),
            ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),
            html.Div('Init Means: ',style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}),
            html.Div(
            dash_table.DataTable(
                id='hmm_means',
                columns=(
                    [{'id': str(p), 'name': str(p)} for p in range(0, 10)]
                ),
                data=[
                    dict(**{str(param): -1 for param in range(0, 10)})
                ],
                style_cell={
                    # all three widths are needed
                    'minWidth': '20px', 'width': '20px', 'maxWidth': '20px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'textAlign': 'center'
                },
                editable = True,
                persistence = True,
                persisted_props = ['data']),
                style = {'width': '50%', 'height' : '50%', 'padding': 5,"margin-left": "60px"}),

            html.Div([
                html.Div('Epoch:'),
                dcc.Input(id = "hmm_epoch", value = 10, type = "number", placeholder="", style={"margin-left": "10px", 'textAlign': 'center', 'width' : '60px'}),
                html.Button('Start', id='hmm_start'),
            ],style={'padding': 5,"margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'})

        ], style = {'width': '90%', 'height' : '20%'}),



        dcc.Tab(label='GMM', value = 'GMM', children=[  
            html.Div([
            html.Div([dcc.Loading(id="loading",type="default", children =dcc.Graph(figure = fig2, id="gmm_hist"))], style={'width': '50%', 'float': 'middle'} ),
            html.Div([  
            html.Div('Fit n Gaussian Peaks: '),
            dcc.RadioItems(
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                '1',
                id = 'gmm_comps',
                labelStyle={'display': 'inline-block', 'marginTop': '5px', "margin-left": "20px"}
            ),
            html.Div('Cov Type: '),
            dcc.RadioItems(
                        ['spherical', 'diag', 'tied'],
                        'spherical',
                        id='gmm_cov_type',
                        labelStyle={'display': 'inline-block', "margin-left": "20px"}
                    ),
            html.Br(),
            html.Div('Init Means: '),
            dash_table.DataTable(
                id='gmm_means',
                columns=(
                    [{'id': str(p), 'name': str(p)} for p in range(0, 10)]
                ),
                data=[
                    dict(**{str(param): -1 for param in range(0, 10)})
                ],
                style_cell={
                    # all three widths are needed
                    'minWidth': '20px', 'width': '20px', 'maxWidth': '20px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'textAlign': 'center'
                },
                editable = True,
                persistence = True,
                persisted_props = ['data']),

            html.Div('Binsize:'),      
            dcc.Slider(
                0.01,
                0.1,
                step=0.01,
                id='binsize',
                value=0.02,
            ),
            html.Br(),
            html.Div([
            dcc.Dropdown(['fret_g', 'fret_b'], 'fret_g', clearable = False, searchable = False, style = {'width' : '100px'}, id = 'gmm_channel'), 
            html.Button('Fit Histogram', id='gmm_fit'),
            html.Button('Save Histogram', id='gmm_save'),
            ],style={'display': 'flex'})
        
            ], style={'width': '30%', 'float': 'middle', 'display': 'inline-block'}),

            ],style={'padding': 5,'width' : "100%",'float': 'middle', 'display': 'flex'}),

        ], style = {'width': '90%', 'height' : '20%'})


    ], vertical = True, style = {'width' : "5%", "height" : '400px'}, content_style = {'width' : "95%", "height" : '100%'}, parent_style ={'width' : "100%", "height" : '100%'}),    






        ##############
       

        html.Div(children = None, id = 'trash', hidden = True),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        
            
            

    ])
    
    return app