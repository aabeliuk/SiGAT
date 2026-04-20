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
            dcc.Graph(id='projection-plot', style={'height': '600px'})
        ])
    ]),
], fluid=True)

@callback(
    Output('projection-plot', 'figure'),
    Input('projection-plot', 'hoverData'),
)
def update_edges_on_hover(hover_data):
    fig = create_base_figure()
    
    if hover_data and 'points' in hover_data and len(hover_data['points']) > 0:
        node_idx = hover_data['points'][0]['customdata']
        
        edge_x, edge_y, edge_colors = [], [], []
        
        # Outgoing positive edges
        for v in adj_lists1_1.get(node_idx, []):
            edge_x += [emb[node_idx, 0], emb[v, 0], None]
            edge_y += [emb[node_idx, 1], emb[v, 1], None]
            edge_colors += ['#27ae60', '#27ae60', None]
        
        # Outgoing negative edges
        for v in adj_lists2_1.get(node_idx, []):
            edge_x += [emb[node_idx, 0], emb[v, 0], None]
            edge_y += [emb[node_idx, 1], emb[v, 1], None]
            edge_colors += ['#c0392b', '#c0392b', None]
        
        # Incoming positive edges
        for u in adj_lists1_2.get(node_idx, []):
            edge_x += [emb[u, 0], emb[node_idx, 0], None]
            edge_y += [emb[u, 1], emb[node_idx, 1], None]
            edge_colors += ['#27ae60', '#27ae60', None]
        
        # Incoming negative edges
        for u in adj_lists2_2.get(node_idx, []):
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
    
    return fig

if __name__ == '__main__':
    print("Starting Dash app on http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop")
    app.run(debug=False, port=8050)
