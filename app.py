from loader import Loader 
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
from layout import make_app
from utils import update_trace, change_trace, select_good_bad, breakpoints_utils, render_good_bad, sl_bkps
from init_fig import init_fig
import logging

logging.getLogger('werkzeug').setLevel(logging.ERROR)
path = ''
fret_g = np.zeros(0)
fret_b = np.zeros(0)
rr = np.zeros(0)
gg = np.zeros(0)
gr = np.zeros(0) 
bb = np.zeros(0)
bg = np.zeros(0)
br = np.zeros(0)
time_r =np.zeros(0)
time_g =np.zeros(0)
time_b = np.zeros(0)
tot_g = gg + gr
tot_b = bb + bg + br
select_list_g = np.zeros(0)
bmode = 1
N_traces = 0
total_frame = 0
idx='N/A'
new = 0
ch_label = 'fret_g'
fret_g_bkps= []
fret_b_bkps= []
b_bkps = []
g_bkps = []
r_bkps = []
bkps = {
        'fret_g' : fret_g_bkps,
        'fret_b' : fret_b_bkps,
        'b' : b_bkps,
        'g' : g_bkps,
        'r' : r_bkps,
    }
time = {
    'fret_b' : time_b,
    'fret_g' : time_g,
    'b' : time_b,
    'g' : time_g,
    'r' : time_r,
}
tot_dtime=[]

    
color=["#fff",'yellow']




fig = init_fig() 

app = make_app()

@app.callback(
    Output('graph', 'figure'),
    Output('i','value'),
    Output('bkp','value'),
    Output('b_bkp','value'),
    Output('AR','value'),
    Output('N_traces','children'),
    Output('set_good','style'),
    Output('set_bad','style'),
    Output('channel', 'options'),
    Output('graph', 'relayoutData'),
    Input('key_events', 'n_events'),
    Input('check', 'value'),
    Input('next', 'n_clicks'),
    Input('previous', 'n_clicks'),
    Input('tr_go', 'n_clicks'),
    Input('dtime','n_clicks'),
    Input('etime','n_clicks'),
    Input('graph', 'clickData'),
    Input('AR','value'),
    Input('save_bkps','n_clicks'),
    Input('load_bkps','n_clicks'),
    Input('loadp', 'n_clicks'),
    Input('rupture','n_clicks'),
    Input('set_good','n_clicks'),
    Input('set_bad','n_clicks'),
    Input('select','n_clicks'),
    Input('scatter', 'value'),
    Input('smooth', 'value'),
    Input('rescale', 'n_clicks'),
    Input("graph", "relayoutData"),
    State('i','value'),
    State('path','value'),
    State('channel', 'value'),
    State('key_events', 'event'),
    )


def update_fig(key_events, check, next, previous, go, dtime, etime, clickData, mode, save, load, loadp, rupture, good, bad, select, scatter, smooth, rescale, relayout, i, path, channel, event):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global N_traces, fig, fig2, idx, total_frame, color, good_style, bad_style, bmode
    global new, time_b, N_traces, total_frame, tot_dtime
    global fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b
    global select_list_g, bkps, ch_label


    if ('n_events' in changed_id) and (event['key'] not in ['q','w','z','x']):
        raise PreventUpdate()
    

    if fig['layout']['uirevision'] == False:
        fig['layout']['uirevision'] = True   

    ##load path##
    if 'loadp' in changed_id:
        fig = init_fig() 
        fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b, N_traces, total_frame, bkps, select_list_g, ch_label = Loader(path).load_traces()
        tot_dtime = []
        i = 0
        new = 1

    ##rescale##
    if 'rescale' in changed_id:
        relayout = {'autosize' : True }
        fig['layout']['uirevision'] = False

    ##change trace##
    i, fig = change_trace(changed_id, event, i, N_traces, fig)
    

    ##select good bad##

    select_list_g = select_good_bad(changed_id, event, i, select_list_g)
    good_style, bad_style = render_good_bad(i, select_list_g)

    #s#ave select##        
    if 'select' in changed_id:  
        np.save(path+r'/selected_g.npy', select_list_g)
     
    ##breakpoints##
    bkps, mode = breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps)
    
    ##save / load breakpoints##
    bkps = sl_bkps(changed_id, path, bkps, mode)

    
    # if 'rupture' in changed_id:
        
    #     rup=Rupture(fret[i])
    #     tot_bkps[i]=rup.det_bkps()      
    #     fig.layout.shapes=[]
    #     fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)
        
    
    ##update trace##
    fig = update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, bkps, smooth)

    ##Display Information##
    if np.any(np.array(bkps['fret_g'], dtype = object)):
        str_g_bkps = ', '.join(str(round(x[1], 2)) for x in bkps['fret_g'][i])
    else:
        str_g_bkps = ''
    if np.any(np.array(bkps['fret_b'], dtype = object)):    
        str_b_bkps = ', '.join(str(round(x[1], 2)) for x in bkps['fret_b'][i])
    else:
        str_b_bkps = ''
    nnote='Total_traces: '+str(N_traces)
    
    return fig, i, str_g_bkps, str_b_bkps, mode, nnote, good_style, bad_style, ch_label, relayout



# @app.callback(
#     Output('Hist','figure'),
#     Input('fitd','n_clicks'),
#     Input('binsize','value'),
#     Input('save_d','n_clicks'),
#     Input('fmode','value'),
#     )
# def update_Dwell(fitd,binsize,saved,fmode):
#     changed_id = [p['prop_id'] for p in callback_context.triggered][0]
#     global fret,  new, select_list, dead_time, tot_bkps, tot_dtime, FRET_list, fig2, exposure_time, xspace, yconvs, yseps, means, weights
#     xspace = np.linspace(0, 1, 1000)
#     if new ==1:
        
#         fig2.data = [list(fig2.data)[0]]
#         FRET_list= calculate_FRET(fret, select_list, tot_bkps)
#         means, cov, weights, yconvs, yseps = calculate_conv(FRET_list)
#         fig2.update_traces(x=FRET_list, selector=dict(name='Dwell_time'))
#         new = 0
        
#     if 'fitd' in changed_id:

#         FRET_list= calculate_FRET(fret, select_list, tot_bkps)
#         means, cov, weights, yconvs, yseps = calculate_conv(FRET_list)
#         fig2.update_traces(x=FRET_list, selector=dict(name='Dwell_time'))
#         fig2.data = [list(fig2.data)[0]]
#         fig2.layout.annotations = []
#         fig2.add_scatter(x = xspace, y=yconvs[int(fmode)-1],marker=dict(color='orange'))
#         for j, sep_gau in enumerate(yseps[int(fmode)-1]):
#             fig2.add_scatter(x=xspace, y=sep_gau, name=f'ysep{j}',marker=dict(color='orange'), line = dict(dash ='dash')  )    
#             fig2.add_annotation(x=means[int(fmode)-1].flatten()[j], y=int(np.max(sep_gau)/2),
#             text= f'{weights[int(fmode)-1][j]*100:.0f}%',
#             showarrow=False,
#             yshift=10)
#     if 'binsize' in changed_id:
#         fig2.update_traces(xbins=dict(start=0,end=1,size=binsize),selector=dict(name='Dwell_time'))
    
#     if 'fmode' in changed_id:
#        fig2.data = [list(fig2.data)[0]]
#        fig2.layout.annotations = []
#        if int(fmode) >0:
#            fig2.add_scatter(x = xspace, y=yconvs[int(fmode)-1],marker=dict(color='orange'))
#            for j, sep_gau in enumerate(yseps[int(fmode)-1]):
#                fig2.add_scatter(x=xspace, y=sep_gau, name=f'ysep{j}',marker=dict(color='orange'), line = dict(dash ='dash')  )      
#                fig2.add_annotation(x=means[int(fmode)-1].flatten()[j], y=int(np.max(sep_gau)/2),
#                text= f'{weights[int(fmode)-1][j]*100:.0f}%',
#                showarrow=False,
#                yshift=10)
#     if 'save_d' in changed_id:
#         if not os.path.exists(path+r"/images"):
#             os.mkdir(path+"/images")
#         fig2.write_image(path+f"/images/hist{fmode}.png", engine="kaleido",width=1600,height=800,scale=10)
    

#     return fig2

    
        
server = app.server 
if __name__ == '__main__':
   app.run_server(debug = False)













