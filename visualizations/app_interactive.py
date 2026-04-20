#!/usr/bin/env python3
"""Interactive Dash app for 2-D projection visualization with hover-based edges."""
import numpy as np
import pickle
from dash import Dash, dcc, html, callback, Input, Output
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# Load pre-computed data
with open("proj_data.pkl", "rb") as f:
    data = pickle.load(f)

emb = data['emb']
idx_to_label = data['idx_to_label']
total_deg = data['total_deg']
pos_ratio = data['pos_ratio']
adj_lists1_1 = data['adj_lists1_1']
adj_lists1_2 = data['adj_lists1_2']
adj_lists2_1 = data['adj_lists2_1']
adj_lists2_2 = data['adj_lists2_2']
N = data['N']

GREEN = '#27ae60'
RED = '#c0392b'

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def create_base_figure():
    """Base figure with nodes but no edges."""
    node_texts = []
    for i in range(N):
        node_texts.append(
            f"<b>{idx_to_label[i]}</b><br>"
            f"Degree: {int(total_deg[i])}<br>"
            f"Positive ratio: {pos_ratio[i]:.2f}<br>"
            f"pos_out={len(adj_lists1_1[i])}, neg_out={len(adj_lists2_1[i])}<br>"
            f"pos_in={len(adj_lists1_2[i])}, neg_in={len(adj_lists2_2[i])}"
        )
    
    return go.Figure(
        data=[go.Scatter(
            x=emb[:, 0],
            y=emb[:, 1],
            mode='markers',
            marker=dict(size=8, color='rgba(52, 73, 94, 0.8)',
                       line=dict(width=0.5, color='#ffffff')),
            text=node_texts,
            hoverinfo='text',
            name='Nodes',
            customdata=list(range(N)),
        )],
        layout=go.Layout(
            title='Interactive 2-D Projection — hover over nodes to see their edges',
            xaxis=dict(title='Proj. Dim 1', zeroline=False),
            yaxis=dict(title='Proj. Dim 2', zeroline=False),
            hovermode='closest',
            showlegend=True,
            margin=dict(l=50, r=20, t=80, b=50),
        )
    )

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Interactive 2-D Node Projection"),
            html.P("Hover over any node to see its connected edges (green=positive, red=negative)")
        ])
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            dcc.Input(
                id='node-search',
                type='text',
                placeholder='Enter node index or ID to search and highlight',
                style={'width': '100%'}
            )
        ], width=6)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Label('Filter by Degree Range:'),
            dcc.RangeSlider(
                id='degree-filter',
                min=0,
                max=int(np.max(total_deg)),
                step=1,
                value=[0, int(np.max(total_deg))],
                marks={0: '0', int(np.max(total_deg)): str(int(np.max(total_deg)))}
            )
        ])
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='projection-plot', style={'height': '600px'})
        ])
    ]),
    dcc.Store(id='hover-store'),
], fluid=True)

@callback(
    Output('hover-store', 'data'),
    Input('projection-plot', 'hoverData'),
)
def update_hover_store(hoverData):
    if (hoverData and 'points' in hoverData and len(hoverData['points']) > 0 and
        hoverData['points'][0].get('curveNumber') == 0 and
        'customdata' in hoverData['points'][0]):
        return hoverData['points'][0]['customdata']
    return None

@callback(
    Output('projection-plot', 'figure'),
    Input('hover-store', 'data'),
    Input('node-search', 'value'),
    Input('degree-filter', 'value'),
)
def update_figure(hover_node, search_value, degree_value):
    min_deg, max_deg = degree_value
    mask = (total_deg >= min_deg) & (total_deg <= max_deg)
    
    # Filtered data
    filtered_emb = emb[mask]
    filtered_customdata = np.arange(N)[mask]  # original indices
    
    node_texts = []
    for orig_i in filtered_customdata:
        node_texts.append(
            f"<b>{idx_to_label[orig_i]}</b><br>"
            f"Degree: {int(total_deg[orig_i])}<br>"
            f"Positive ratio: {pos_ratio[orig_i]:.2f}<br>"
            f"pos_out={len(adj_lists1_1[orig_i])}, neg_out={len(adj_lists2_1[orig_i])}<br>"
            f"pos_in={len(adj_lists1_2[orig_i])}, neg_in={len(adj_lists2_2[orig_i])}"
        )
    
    fig = go.Figure(
        data=[go.Scatter(
            x=filtered_emb[:, 0],
            y=filtered_emb[:, 1],
            mode='markers',
            marker=dict(size=8, color='rgba(52, 73, 94, 0.8)',
                       line=dict(width=0.5, color='#ffffff')),
            text=node_texts,
            hoverinfo='text',
            name='Nodes',
            customdata=list(filtered_customdata),
        )],
        layout=go.Layout(
            title='Interactive 2-D Projection — hover over nodes to see their edges',
            xaxis=dict(title='Proj. Dim 1', zeroline=False),
            yaxis=dict(title='Proj. Dim 2', zeroline=False),
            hovermode='closest',
            showlegend=True,
            margin=dict(l=50, r=20, t=80, b=50),
        )
    )
    
    node_idx = None
    highlight_node = False
    
    if search_value:
        if search_value.isdigit():
            node_idx = int(search_value)
            if 0 <= node_idx < N and mask[node_idx]:
                highlight_node = True
            else:
                return fig  # invalid or filtered out
        else:
            # search by label
            if search_value in idx_to_label.values():
                node_idx = list(idx_to_label.keys())[list(idx_to_label.values()).index(search_value)]
                if mask[node_idx]:
                    highlight_node = True
                else:
                    return fig  # filtered out
            else:
                return fig  # invalid label
    elif hover_node is not None:
        node_idx = hover_node
    
    if node_idx is not None:
        
        edge_x, edge_y, edge_colors = [], [], []
        
        # Outgoing positive edges
        for v in adj_lists1_1.get(node_idx, []):
            if mask[v]:
                edge_x += [emb[node_idx, 0], emb[v, 0], None]
                edge_y += [emb[node_idx, 1], emb[v, 1], None]
                edge_colors += ['#27ae60', '#27ae60', None]
        
        # Outgoing negative edges
        for v in adj_lists2_1.get(node_idx, []):
            if mask[v]:
                edge_x += [emb[node_idx, 0], emb[v, 0], None]
                edge_y += [emb[node_idx, 1], emb[v, 1], None]
                edge_colors += ['#c0392b', '#c0392b', None]
        
        # Incoming positive edges
        for u in adj_lists1_2.get(node_idx, []):
            if mask[u]:
                edge_x += [emb[u, 0], emb[node_idx, 0], None]
                edge_y += [emb[u, 1], emb[node_idx, 1], None]
                edge_colors += ['#27ae60', '#27ae60', None]
        
        # Incoming negative edges
        for u in adj_lists2_2.get(node_idx, []):
            if mask[u]:
                edge_x += [emb[u, 0], emb[node_idx, 0], None]
                edge_y += [emb[u, 1], emb[node_idx, 1], None]
                edge_colors += ['#c0392b', '#c0392b', None]
        
        if edge_x:
            pos_edge_x, pos_edge_y = [], []
            neg_edge_x, neg_edge_y = [], []
            
            for i in range(0, len(edge_x), 3):
                if edge_colors[i] == '#27ae60':
                    pos_edge_x += [edge_x[i], edge_x[i+1], None]
                    pos_edge_y += [edge_y[i], edge_y[i+1], None]
                else:
                    neg_edge_x += [edge_x[i], edge_x[i+1], None]
                    neg_edge_y += [edge_y[i], edge_y[i+1], None]
            
            if pos_edge_x:
                fig.add_trace(go.Scatter(
                    x=pos_edge_x, y=pos_edge_y, mode='lines',
                    line=dict(color='#27ae60', width=1.5),
                    hoverinfo='none', name='Positive edges'
                ))
            if neg_edge_x:
                fig.add_trace(go.Scatter(
                    x=neg_edge_x, y=neg_edge_y, mode='lines',
                    line=dict(color='#c0392b', width=1.5),
                    hoverinfo='none', name='Negative edges'
                ))
    
    if highlight_node:
        fig.add_trace(go.Scatter(
            x=[emb[node_idx, 0]],
            y=[emb[node_idx, 1]],
            mode='markers',
            marker=dict(size=12, color='yellow', line=dict(width=2, color='black')),
            text=[f"<b>SEARCHED: {idx_to_label[node_idx]}</b><br>" + fig.data[0].text[list(filtered_customdata).index(node_idx)].split('<br>', 1)[1]],
            hoverinfo='text',
            name='Searched Node',
            showlegend=True
        ))
        fig.update_layout(title='Interactive 2-D Projection — searched node highlighted with edges')
    
    return fig

if __name__ == '__main__':
    print("Starting Dash app on http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop")
    app.run(debug=False, port=8050)
