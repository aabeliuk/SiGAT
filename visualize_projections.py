#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_projections.py
========================
Visualises the Orthogonal Projection Layer embeddings for the Chilean
political sentiment network (edgelist_v3_2025H2_2026H1.csv).

All model classes (SDGNN, Encoder, OrthogonalProjectionLayer,
AttentionAggregator / MeanAggregator) and the training step are
imported directly from sdgnn.py — no model code is duplicated here.

Flow
----
1. Load & parse the xlsx edgelist.
2. Configure sdgnn module globals for this dataset.
3. Build the model using sdgnn.Encoder + sdgnn.SDGNN (same structure as
   sdgnn.run() but for the xlsx graph).
4. Train with model.criterion() + model.orthogonality_loss() — both
   defined in sdgnn.SDGNN.
5. Extract projected embeddings via model.forward() and visualise.

Outputs (./visualizations/)
---------------------------
  proj_scatter.png       – 2-D scatter: positive-ratio, degree, binary
  proj_edges.png         – Sampled signed edges as arrows
  training_curves.png    – Task loss + orthogonality loss per epoch
  sentiment_analysis.png – Per-node positive-ratio ranking & distribution
  projection_matrix.png  – Heatmap of learned W and W W^T

Usage
-----
    # conda activate torch_env
    python visualize_projections.py
    python visualize_projections.py --proj_dim 3 --epochs 60 --agg mean
"""

import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
import torch.nn as nn

# ── Import everything from sdgnn.py — no model code lives here ────────────
import sdgnn
from sdgnn import (
    SDGNN,
    Encoder,
    OrthogonalProjectionLayer,   # noqa: F401  (used indirectly via SDGNN)
    AttentionAggregator,
    MeanAggregator,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLI  (own parser — does NOT conflict with sdgnn.parser)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(
    description='Visualise SDGNN Orthogonal Projections for the  edgelist')
parser.add_argument('--csv',
    default='experiment-data/edgelist_v3_2025H2_2026H1.csv')
parser.add_argument('--out_dir',      default='visualizations')
parser.add_argument('--proj_dim',     type=int,   default=2,
    help='Dimension of the orthogonal projection (2 → 2-D, ≥3 → also 3-D)')
parser.add_argument('--embed_dim',    type=int,   default=20,
    help='Encoder embedding dimension (EMBEDDING_SIZE1 in sdgnn)')
parser.add_argument('--feat_dim',     type=int,   default=20,
    help='Node feature dimension (NODE_FEAT_SIZE in sdgnn)')
parser.add_argument('--epochs',       type=int,   default=50)
parser.add_argument('--batch_size',   type=int,   default=128)
parser.add_argument('--lr',           type=float, default=5e-3)
parser.add_argument('--ortho_weight', type=float, default=5.0,
    help='λ for L_o = ||WW^T - I||_F^2  (Bousmalis et al., 2016). '
         'Scale relative to task loss: with task loss ~500 and L_o ~6, '
         'λ≈5–15 keeps the penalty at roughly 10%% of the total loss.')
parser.add_argument('--agg',          default='attention',
    choices=['mean', 'attention'],
    help='Aggregator type — mirrors sdgnn.py --agg')
parser.add_argument('--label_top',    type=int,   default=200)
parser.add_argument('--max_edges',    type=int,   default=400)
parser.add_argument('--min_degree',   type=int,   default=5)
parser.add_argument('--seed',         type=int,   default=42)
parser.add_argument('--device',       default='cpu')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)

# ── Configure sdgnn module globals so its classes use our settings ─────────
# This mirrors what sdgnn.main() does before calling sdgnn.run().
sdgnn.DEVICES         = torch.device(args.device)
sdgnn.NODE_FEAT_SIZE  = args.feat_dim
sdgnn.EMBEDDING_SIZE1 = args.embed_dim
sdgnn.PROJ_DIM        = args.proj_dim
sdgnn.ORTHO_WEIGHT    = args.ortho_weight
sdgnn.BATCH_SIZE      = args.batch_size
sdgnn.EPOCHS          = args.epochs
sdgnn.LEARNING_RATE   = args.lr
sdgnn.DROUPOUT        = 0.0

DEVICE = sdgnn.DEVICES

for _style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid', 'ggplot'):
    try:
        plt.style.use(_style)
        break
    except OSError:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 1 · Load  edgelist
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Loading {args.csv} …")
df = pd.read_csv(args.csv)
print(f"  {len(df):,} rows")

FROM_COL = 'from_node_label'
TO_COL   = 'to_node_label'
SIGN_COL = 'sentiment'

sign_map = {'positive': 1, 'negative': -1}
df[FROM_COL] = df[FROM_COL].astype(str).str.strip()
df[TO_COL]   = df[TO_COL].astype(str).str.strip()
df = df[df[SIGN_COL].isin(sign_map)].copy()
df['sign'] = df[SIGN_COL].map(sign_map).astype(int)
df = df[df[FROM_COL] != df[TO_COL]]      # remove self-loops
print(f"  After filtering: {len(df):,} edges "
      f"({df['sign'].eq(1).sum():,} pos, {df['sign'].eq(-1).sum():,} neg)")

# Integer node mapping  (politician label → integer ID)
all_labels   = sorted(set(df[FROM_COL]) | set(df[TO_COL]))
N            = len(all_labels)
node_map     = {name: i for i, name in enumerate(all_labels)}
idx_to_label = {i: name for name, i in node_map.items()}
print(f"  {N} unique nodes")

# Filter out nodes with degree <= 2 before embedding.
# Keep only nodes that remain in the 2-core of the signed graph.
keep_labels = set(all_labels)
while True:
    deg_counts = defaultdict(int)
    for _, row in df[df[FROM_COL].isin(keep_labels) &
                    df[TO_COL].isin(keep_labels)].iterrows():
        deg_counts[row[FROM_COL]] += 1
        deg_counts[row[TO_COL]] += 1

    to_remove = {label for label in keep_labels if deg_counts[label] <= 2}
    if not to_remove:
        break
    keep_labels -= to_remove

if len(keep_labels) != len(all_labels):
    print(f"Filtering {len(all_labels) - len(keep_labels)} nodes with degree <= 2 before embedding")
    df = df[df[FROM_COL].isin(keep_labels) & df[TO_COL].isin(keep_labels)].copy()
    all_labels = sorted(keep_labels)
    N = len(all_labels)
    node_map = {name: i for i, name in enumerate(all_labels)}
    idx_to_label = {i: name for name, i in node_map.items()}
    print(f"  {N} unique nodes remain after filtering")

# ─────────────────────────────────────────────────────────────────────────────
# 2 · Build adjacency structures (same format as sdgnn.load_data2 / sdgnn.run)
#
#   adj_lists1_1 : directed positive  (u → v) — Encoder aggregation type 0
#   adj_lists1_2 : directed positive  (v ← u) — Encoder aggregation type 1
#   adj_lists2_1 : directed negative  (u → v) — Encoder aggregation type 2
#   adj_lists2_2 : directed negative  (v ← u) — Encoder aggregation type 3
#   adj_lists1   : undirected positive — pos_neighbors in SDGNN.criterion
#   adj_lists2   : undirected negative — neg_neighbors in SDGNN.criterion
#   weight_dict  : motif weights (1.0 — FeaExtra is dataset-specific)
# ─────────────────────────────────────────────────────────────────────────────
print("Building adjacency structures …")
adj_lists1   = defaultdict(set)
adj_lists1_1 = defaultdict(set)
adj_lists1_2 = defaultdict(set)
adj_lists2   = defaultdict(set)
adj_lists2_1 = defaultdict(set)
adj_lists2_2 = defaultdict(set)
weight_dict  = defaultdict(dict)

for _, row in df.iterrows():
    u = node_map[row[FROM_COL]]
    v = node_map[row[TO_COL]]
    s = int(row['sign'])
    if s == 1:
        adj_lists1[u].add(v);   adj_lists1[v].add(u)
        adj_lists1_1[u].add(v); adj_lists1_2[v].add(u)
    else:
        adj_lists2[u].add(v);   adj_lists2[v].add(u)
        adj_lists2_1[u].add(v); adj_lists2_2[v].add(u)

# Uniform motif weights (FeaExtra not available for this dataset)
for u in range(N):
    for v in adj_lists1_1[u]:
        weight_dict[u][v] = 1.0
    for v in adj_lists2_1[u]:
        weight_dict[u][v] = 1.0

# Convert directed adj_lists → scipy sparse matrices for sdgnn.Encoder
# Same transformation as sdgnn.run()'s inner `func` helper.
def _to_csr(adj_dict: dict, n: int) -> sp.csr_matrix:
    edges = [(u, v) for u, vs in adj_dict.items() for v in vs]
    if not edges:
        return sp.csr_matrix((n, n))
    rows, cols = zip(*edges)
    return sp.csr_matrix(
        (np.ones(len(edges)), (rows, cols)), shape=(n, n))

adj_sparse = [_to_csr(d, N)
              for d in (adj_lists1_1, adj_lists1_2, adj_lists2_1, adj_lists2_2)]

# Per-node statistics for visualisation
pod       = np.array([len(adj_lists1_1[i]) for i in range(N)])
nod       = np.array([len(adj_lists2_1[i]) for i in range(N)])
pid       = np.array([len(adj_lists1_2[i]) for i in range(N)])
nid       = np.array([len(adj_lists2_2[i]) for i in range(N)])
total_deg = pod + nod + pid + nid
pos_ratio = (pod + pid) / np.where(total_deg > 0, total_deg, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 3 · Build model — identical structure to sdgnn.run(), using sdgnn classes
#
#   sdgnn.run() builds:
#       features (nn.Embedding)
#       → enc1 (sdgnn.Encoder, layer 1)
#       → enc2 (sdgnn.Encoder, layer 2, wraps enc1 as a lambda)
#       → sdgnn.SDGNN(enc2)   ← contains sdgnn.OrthogonalProjectionLayer
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Building model  (feat={args.feat_dim}, embed={args.embed_dim}, "
      f"proj={args.proj_dim}, agg={args.agg}) …")

num_nodes_model = N + 3          # same "+3" buffer used in sdgnn.run()

features = nn.Embedding(num_nodes_model, sdgnn.NODE_FEAT_SIZE)
features.weight.requires_grad = True
features = features.to(DEVICE)

aggregator_cls = AttentionAggregator if args.agg == 'attention' else MeanAggregator

# Layer-1 encoder (mirrors sdgnn.run())
aggs1 = [aggregator_cls(features, sdgnn.NODE_FEAT_SIZE,
                         sdgnn.NODE_FEAT_SIZE, num_nodes_model)
          for _ in adj_sparse]
enc1  = Encoder(features, sdgnn.NODE_FEAT_SIZE,
                sdgnn.EMBEDDING_SIZE1, adj_sparse, aggs1)
enc1  = enc1.to(DEVICE)

# Layer-2 encoder wraps enc1 via lambda — identical to sdgnn.run()
aggs2 = [aggregator_cls(lambda n: enc1(n), sdgnn.EMBEDDING_SIZE1,
                         sdgnn.EMBEDDING_SIZE1, num_nodes_model)
          for _ in adj_sparse]
enc2  = Encoder(lambda n: enc1(n), sdgnn.EMBEDDING_SIZE1,
                sdgnn.EMBEDDING_SIZE1, adj_sparse, aggs2)

# SDGNN with OrthogonalProjectionLayer — both defined in sdgnn.py
model = SDGNN(enc2, proj_dim=args.proj_dim)
model = model.to(DEVICE)

total_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in enc1.parameters()) +
                sum(p.numel() for p in features.parameters()))
print(f"Total parameters : {total_params:,}")

# Optimizer — same setup as sdgnn.run()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad,
           list(model.parameters()) +
           list(enc1.parameters())),
    lr=sdgnn.LEARNING_RATE,
    weight_decay=0.001,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4 · Training loop
#     model.criterion()         — sdgnn.SDGNN method, called unmodified
#     model.orthogonality_loss()— sdgnn.SDGNN method, called unmodified
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nTraining {args.epochs} epochs …")
all_nodes    = list(range(N))
task_losses  : list = []
ortho_losses : list = []

model.train()
for epoch in range(1, args.epochs + 1):
    random.shuffle(all_nodes)
    ep_loss = 0.0
    n_steps = 0

    for i in range(0, N, args.batch_size):
        batch = all_nodes[i: i + args.batch_size]
        optimizer.zero_grad()

        # sdgnn.SDGNN.criterion — unchanged, called with the csv adj structures
        loss = model.criterion(
            batch,
            adj_lists1,    # undirected pos  → pos_neighbors
            adj_lists2,    # undirected neg  → neg_neighbors
            adj_lists1_1,  # directed pos    → direction / triangle loss
            adj_lists2_1,  # directed neg    → direction / triangle loss
            weight_dict,
        )
        # sdgnn.SDGNN.orthogonality_loss — unchanged
        loss = loss + sdgnn.ORTHO_WEIGHT * model.orthogonality_loss()

        loss.backward()
        optimizer.step()
        ep_loss += loss.item()
        n_steps += 1

    ep_loss /= max(n_steps, 1)
    o_loss   = model.orthogonality_loss().item()
    task_losses.append(ep_loss)
    ortho_losses.append(o_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"loss={ep_loss:.4f}   L_o={o_loss:.6f}")

print("Training complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 5 · Extract projected embeddings via model.forward() — sdgnn.SDGNN method
#     forward() calls self.enc then self.proj; both defined in sdgnn.py
# ─────────────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    emb = model.forward(list(range(N))).cpu().numpy()   # (N, proj_dim)

W_np      = model.proj.projection.weight.detach().cpu().numpy()   # (proj_dim, embed_dim)
WWT       = W_np @ W_np.T
I_        = np.eye(W_np.shape[0])
ortho_err = float(np.linalg.norm(WWT - I_, 'fro'))
print(f"\nEmbeddings : {emb.shape}  |  ||WW^T - I||_F = {ortho_err:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for all figures
# ─────────────────────────────────────────────────────────────────────────────
GREEN = '#27ae60'; RED = '#c0392b'; GRAY = '#bdc3c7'
CMAP_SIGN = 'RdYlGn'; CMAP_DEG = 'viridis'
norm_ratio = Normalize(vmin=0, vmax=1)
log_deg    = np.log1p(total_deg)
is_pos     = pos_ratio > 0.5


def _short(name: str, maxlen: int = 20) -> str:
    if len(name) <= maxlen:
        return name
    parts = name.split()
    return (parts[0] + ' ' + parts[-1]) if len(parts) > 1 else name[:maxlen]


def _annotate_top(ax, xy, deg, top_n, fs=6):
    for i in np.argsort(deg)[-top_n:]:
        ax.annotate(_short(idx_to_label[i]), (xy[i, 0], xy[i, 1]),
                    fontsize=fs, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points',
                    color='#2c3e50')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — proj_scatter.png  (3 colour-coded 2-D scatter panels)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSaving figures → {args.out_dir}/")
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
fig.suptitle(
    'SDGNN Orthogonal Projection — Chilean Political Sentiment Network\n'
    f'proj_dim={args.proj_dim}  embed_dim={args.embed_dim}  '
    f'{args.epochs} epochs  {N} nodes  {len(df):,} signed edges',
    fontsize=12, fontweight='bold', y=1.01)

ax = axes[0]
sc = ax.scatter(emb[:, 0], emb[:, 1],
                c=pos_ratio, cmap=CMAP_SIGN, norm=norm_ratio,
                s=35, alpha=0.85, linewidths=0.3, edgecolors='#7f8c8d')
plt.colorbar(sc, ax=ax, label='Positive Edge Ratio', shrink=0.8, pad=0.01)
ax.set_title('Positive Edge Ratio\n(green = positive sentiment)', fontsize=10)
ax.set_xlabel('Proj. Dim 1'); ax.set_ylabel('Proj. Dim 2')
_annotate_top(ax, emb, total_deg, args.label_top)

ax = axes[1]
sc = ax.scatter(emb[:, 0], emb[:, 1], c=log_deg, cmap=CMAP_DEG,
                s=35, alpha=0.85, linewidths=0.3, edgecolors='#7f8c8d')
plt.colorbar(sc, ax=ax, label='log(1 + degree)', shrink=0.8, pad=0.01)
ax.set_title('Node Degree\n(bright = high connectivity)', fontsize=10)
ax.set_xlabel('Proj. Dim 1'); ax.set_ylabel('Proj. Dim 2')
_annotate_top(ax, emb, total_deg, args.label_top)

ax = axes[2]
ax.scatter(emb[is_pos,  0], emb[is_pos,  1], c=GREEN, s=35, alpha=0.8,
           linewidths=0.3, edgecolors='#7f8c8d',
           label=f'Mostly positive ({is_pos.sum()})')
ax.scatter(emb[~is_pos, 0], emb[~is_pos, 1], c=RED, s=35, alpha=0.8,
           linewidths=0.3, edgecolors='#7f8c8d',
           label=f'Mostly negative ({(~is_pos).sum()})')
ax.legend(fontsize=9, framealpha=0.9)
ax.set_title('Sentiment Profile\n(green=pos-majority, red=neg-majority)',
             fontsize=10)
ax.set_xlabel('Proj. Dim 1'); ax.set_ylabel('Proj. Dim 2')
_annotate_top(ax, emb, total_deg, args.label_top)

plt.tight_layout()
p = os.path.join(args.out_dir, 'proj_scatter.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1b — proj_extreme.png  (highest-degree nodes)
# ─────────────────────────────────────────────────────────────────────────────
def _highest_degree(indices, deg, top_n=15):
    """Return the top-n indices from the provided set ranked by degree."""
    order = np.argsort(deg[indices])
    return indices[order[-top_n:]]

eligible = np.where(total_deg >= args.min_degree)[0]
if len(eligible) == 0:
    eligible = np.arange(N)
extreme_idx = _highest_degree(eligible, total_deg,
                                   top_n=min(100, args.label_top))

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor('#f8f9fa')
ax.scatter(emb[:, 0], emb[:, 1], c=GRAY, s=24, alpha=0.6, linewidths=0)
ax.scatter(emb[extreme_idx, 0], emb[extreme_idx, 1],
           c='#34495e', s=90, edgecolors='white', linewidths=1.2)
for i in extreme_idx:
    ax.annotate(_short(idx_to_label[i], maxlen=18),
                (emb[i, 0], emb[i, 1]),
                fontsize=8, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points',
                color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.75,
                          ec='#7f8c8d', lw=0.5))
ax.set_title(
    'Extreme Nodes in 2-D Projection Space\n'
    '(nodes with largest 2-D embedding magnitude, degree ≥ min_degree)',
    fontsize=12)
ax.set_xlabel('Proj. Dim 1'); ax.set_ylabel('Proj. Dim 2')
ax.grid(True, alpha=0.25)
plt.tight_layout()
p = os.path.join(args.out_dir, 'proj_extreme.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1c — Interactive Dash app  (hover edges, Plotly)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    import pickle
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from dash import Dash, dcc, html, callback, Input, Output
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

if PLOTLY_AVAILABLE:
    print(f"Generating interactive HTML scatter → {args.out_dir}/proj_scatter_interactive.html")
    scatter_html = go.Figure(
        data=[go.Scatter(
            x=emb[:, 0],
            y=emb[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=pos_ratio,
                colorscale='RdYlGn',
                colorbar=dict(title='Positive Edge Ratio'),
                line=dict(width=0.3, color='#7f8c8d'),
            ),
            text=[
                f"<b>{idx_to_label[i]}</b><br>degree={int(total_deg[i])}<br>"
                f"pos_ratio={pos_ratio[i]:.2f}"
                for i in range(N)
            ],
            hoverinfo='text',
        )],
        layout=go.Layout(
            title='Interactive 2-D Projection — Positive Edge Ratio',
            xaxis=dict(title='Proj. Dim 1'),
            yaxis=dict(title='Proj. Dim 2'),
            hovermode='closest',
        )
    )
    html_path = os.path.join(args.out_dir, 'proj_scatter_interactive.html')
    pio.write_html(scatter_html, file=html_path,
                   full_html=True, include_plotlyjs='cdn')
    print(f"  ✓  {html_path}")

    if DASH_AVAILABLE:
        print(f"Generating interactive Dash app → {args.out_dir}/app_interactive.py")
        
        dash_app_code = '''#!/usr/bin/env python3
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
'''
        
        app_file = os.path.join(args.out_dir, 'app_interactive.py')
        with open(app_file, 'w') as f:
            f.write(dash_app_code)
        
        # Save data for the Dash app
        data_to_save = {
            'emb': emb,
            'idx_to_label': idx_to_label,
            'total_deg': total_deg,
            'pos_ratio': pos_ratio,
            'adj_lists1_1': adj_lists1_1,
            'adj_lists1_2': adj_lists1_2,
            'adj_lists2_1': adj_lists2_1,
            'adj_lists2_2': adj_lists2_2,
            'N': N,
        }
        data_file = os.path.join(args.out_dir, 'proj_data.pkl')
        with open(data_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"  ✓  {app_file}")
        print(f"  ✓  {data_file}")
        print(f"\n  To run the interactive app:")
        print(f"    cd {args.out_dir}")
        print(f"    python app_interactive.py")
    else:
        print("Dash not installed; skipping interactive Dash app.")
else:
    print("Plotly not installed; skipping interactive HTML scatter and Dash app.")

edges_list = [
    (node_map[row[FROM_COL]], node_map[row[TO_COL]], int(row['sign']))
    for _, row in df.iterrows()
]
sample_edges = random.sample(edges_list, min(args.max_edges, len(edges_list)))

fig, ax = plt.subplots(figsize=(11, 9))
ax.set_facecolor('#f8f9fa')
ax.scatter(emb[:, 0], emb[:, 1], c=GRAY, s=18, alpha=0.45, zorder=1, linewidths=0)
for u, v, sign in sample_edges:
    ax.annotate("",
                xy=(emb[v, 0], emb[v, 1]), xytext=(emb[u, 0], emb[u, 1]),
                arrowprops=dict(arrowstyle="-|>",
                                color=GREEN if sign == 1 else RED,
                                alpha=0.30, lw=0.65, mutation_scale=7),
                zorder=2)
top_e     = min(args.label_top, 20)
top_e_idx = np.argsort(total_deg)[-top_e:]
ax.scatter(emb[top_e_idx, 0], emb[top_e_idx, 1],
           c='#2c3e50', s=60, zorder=3, linewidths=0)
_annotate_top(ax, emb, total_deg, top_e, fs=7)
ax.legend(handles=[
    Line2D([0], [0], color=GREEN, lw=2, label='Positive sentiment'),
    Line2D([0], [0], color=RED,   lw=2, label='Negative sentiment'),
], loc='upper right', fontsize=10, framealpha=0.9)
ax.set_title(
    f'Signed Edges in Orthogonal Projection Space\n'
    f'(sample {len(sample_edges):,} / {len(edges_list):,} edges, '
    f'top-{top_e} nodes highlighted)', fontsize=11)
ax.set_xlabel('Proj. Dim 1'); ax.set_ylabel('Proj. Dim 2')
plt.tight_layout()
p = os.path.join(args.out_dir, 'proj_edges.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — training_curves.png
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle('Training Dynamics', fontsize=12, fontweight='bold')
ep_ax = range(1, args.epochs + 1)
ax1.plot(ep_ax, task_losses, color='steelblue', lw=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Task Loss per Epoch\n(sign BCE + λ · L_o)'); ax1.grid(True, alpha=0.3)
ax2.plot(ep_ax, ortho_losses, color='#e74c3c', lw=2)
ax2.axhline(y=0, color='#2c3e50', ls='--', alpha=0.6,
            label='Perfect orthogonality (L_o = 0)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel(r'$\|WW^\top - I\|_F^2$')
ax2.set_title(r'Orthogonality Penalty $L_o$' +
              '\n(Bousmalis et al., 2016 · Brock et al., 2017)')
ax2.legend(fontsize=9, framealpha=0.9); ax2.grid(True, alpha=0.3)
plt.tight_layout()
p = os.path.join(args.out_dir, 'training_curves.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — sentiment_analysis.png
# ─────────────────────────────────────────────────────────────────────────────
active_mask   = total_deg >= args.min_degree
active_idx    = np.where(active_mask)[0]
active_labels = [idx_to_label[i] for i in active_idx]
active_ratios = pos_ratio[active_idx]
active_deg    = total_deg[active_idx]
sort_ord      = np.argsort(active_ratios)
NBAR          = 15

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    f'Node-Level Sentiment Analysis  (nodes with degree ≥ {args.min_degree})\n'
    f'{len(active_idx)} / {N} nodes shown',
    fontsize=12, fontweight='bold')

ax = axes[0]
b_idx = sort_ord[:NBAR]
ax.barh(range(NBAR), active_ratios[b_idx],
        color=RED, alpha=0.85, edgecolor='white', lw=0.5)
ax.set_yticks(range(NBAR))
ax.set_yticklabels([_short(active_labels[i], 22) for i in b_idx], fontsize=8)
ax.set_xlabel('Positive Edge Ratio')
ax.set_title(f'Top-{NBAR} Most Negatively Perceived', fontsize=10)
ax.set_xlim(0, 1); ax.axvline(x=0.5, color='#7f8c8d', ls='--', alpha=0.6, lw=1)
for j, i in enumerate(b_idx):
    ax.text(active_ratios[i] + 0.01, j, f"deg={active_deg[i]}",
            va='center', fontsize=7, color='#555')

ax = axes[1]
t_idx = sort_ord[-NBAR:][::-1]
ax.barh(range(NBAR), active_ratios[t_idx],
        color=GREEN, alpha=0.85, edgecolor='white', lw=0.5) 
ax.set_yticks(range(NBAR))
ax.set_yticklabels([_short(active_labels[i], 22) for i in t_idx], fontsize=8)
ax.set_xlabel('Positive Edge Ratio')
ax.set_title(f'Top-{NBAR} Most Positively Perceived', fontsize=10)
ax.set_xlim(0, 1); ax.axvline(x=0.5, color='#7f8c8d', ls='--', alpha=0.6, lw=1)
for j, i in enumerate(t_idx):
    ax.text(max(active_ratios[i] - 0.02, 0.01), j,
            f"deg={active_deg[i]}", va='center', ha='right',
            fontsize=7, color='#555')

ax = axes[2]
ax.hist(active_ratios, bins=25, color='steelblue', alpha=0.8, edgecolor='white')
ax.axvline(active_ratios.mean(), color='darkorange', ls='--', lw=2,
           label=f'Mean = {active_ratios.mean():.3f}')
ax.axvline(0.5, color='#7f8c8d', ls=':', lw=1.5, label='Neutral = 0.5')
ax.set_xlabel('Positive Edge Ratio'); ax.set_ylabel('Node Count')
ax.set_title(f'Distribution (n = {len(active_idx)} nodes)')
ax.legend(fontsize=9, framealpha=0.9); ax.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(args.out_dir, 'sentiment_analysis.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — projection_matrix.png  (W heatmap + W W^T Gram matrix)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5 + args.proj_dim * 0.3))
fig.suptitle(
    f'Learned Orthogonal Projection Matrix W  '
    f'({args.proj_dim} × {args.embed_dim})\n'
    r'$\|WW^\top - I\|_F$ = ' + f'{ortho_err:.5f}  '
    r'(0 = perfectly orthogonal rows)',
    fontsize=11, fontweight='bold')

ax  = axes[0]
im  = ax.imshow(W_np, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Weight', shrink=0.8)
ax.set_xlabel('Embedding Dimension'); ax.set_ylabel('Projection Dimension')
ax.set_yticks(range(args.proj_dim))
ax.set_yticklabels([f'Proj {i+1}' for i in range(args.proj_dim)])
ax.set_title('W  (projection weight)')

ax  = axes[1]
im2 = ax.imshow(WWT, cmap='coolwarm', aspect='auto', vmin=-0.2, vmax=1.2)
plt.colorbar(im2, ax=ax, label='Value', shrink=0.8)
ax.set_xticks(range(args.proj_dim)); ax.set_yticks(range(args.proj_dim))
ax.set_xticklabels([f'Proj {i+1}' for i in range(args.proj_dim)])
ax.set_yticklabels([f'Proj {i+1}' for i in range(args.proj_dim)])
ax.set_title(r'$WW^\top$  (should be $\approx I$)')
for i in range(args.proj_dim):
    for j in range(args.proj_dim):
        ax.text(j, i, f'{WWT[i, j]:.3f}', ha='center', va='center',
                fontsize=8,
                color='white' if abs(WWT[i, j] - 0.7) > 0.4 else '#2c3e50')

plt.tight_layout()
p = os.path.join(args.out_dir, 'projection_matrix.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — proj_angular.png
#
# The sign loss uses the dot product  z_u · z_v  as the prediction logit, so
# the geometry is ANGULAR (inner-product), not Euclidean.
# Positive pairs should be directionally aligned (cos θ > 0);
# negative pairs should point in roughly opposite directions (cos θ < 0).
#
# Three panels (2-D case):
#   (a) Unit circle — nodes projected to unit sphere, edges as chords
#   (b) Cosine similarity histogram — P(positive) vs P(negative) | cos θ
#   (c) Polar angle θ vs positive-ratio scatter (only proj_dim == 2)
#
# For proj_dim > 2 only panel (b) is produced.
# ─────────────────────────────────────────────────────────────────────────────
emb_norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
emb_unit  = emb / emb_norms          # (N, proj_dim) unit vectors

# Compute per-edge cosine similarities
cos_pos, cos_neg = [], []
for u, v, sign in edges_list:
    c = float(np.dot(emb_unit[u], emb_unit[v]))
    if sign == 1:
        cos_pos.append(c)
    else:
        cos_neg.append(c)
cos_pos = np.array(cos_pos)
cos_neg = np.array(cos_neg)

if args.proj_dim == 2:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        'Angular Geometry of the Orthogonal Projection Space\n'
        r'Sign loss: $\hat{y}_{uv} = \sigma(z_u \cdot z_v)$  — '
        r'alignment ($\cos\theta > 0$) $\Rightarrow$ positive, '
        r'opposition ($\cos\theta < 0$) $\Rightarrow$ negative',
        fontsize=11, fontweight='bold', y=1.02)

    # ── (a) Unit circle ──────────────────────────────────────────────────────
    ax = axes[0]
    theta_circle = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle),
            color='#bdc3c7', lw=1.2, zorder=0)

    # Sample edges as chords
    n_chord = min(400, len(edges_list))
    sample_chord = random.sample(edges_list, n_chord)
    for u, v, sign in sample_chord:
        ax.plot([emb_unit[u, 0], emb_unit[v, 0]],
                [emb_unit[u, 1], emb_unit[v, 1]],
                color=GREEN if sign == 1 else RED,
                alpha=0.12, lw=0.7, zorder=1)

    # Nodes coloured by positive-ratio
    sc = ax.scatter(emb_unit[:, 0], emb_unit[:, 1],
                    c=pos_ratio, cmap=CMAP_SIGN, norm=norm_ratio,
                    s=28, alpha=0.85, linewidths=0.3,
                    edgecolors='#7f8c8d', zorder=2)
    plt.colorbar(sc, ax=ax, label='Positive ratio', shrink=0.75, pad=0.02)

    # Label highest-degree nodes
    for i in np.argsort(total_deg)[-min(15, args.label_top):]:
        ax.annotate(_short(idx_to_label[i]),
                    (emb_unit[i, 0], emb_unit[i, 1]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', color='#2c3e50')

    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
    ax.set_title('Unit Circle — nodes normalised to ‖z‖ = 1\n'
                 'Chords: green = positive, red = negative', fontsize=10)
    ax.set_xlabel('Proj. Dim 1 (unit)'); ax.set_ylabel('Proj. Dim 2 (unit)')
    ax.legend(handles=[
        Line2D([0], [0], color=GREEN, lw=2, label='Positive chord'),
        Line2D([0], [0], color=RED,   lw=2, label='Negative chord'),
    ], fontsize=8, loc='upper right', framealpha=0.9)

    # ── (b) Cosine similarity histogram ──────────────────────────────────────
    ax = axes[1]
    bins = np.linspace(-1, 1, 41)
    ax.hist(cos_pos, bins=bins, color=GREEN, alpha=0.65,
            label=f'Positive edges  (n={len(cos_pos):,})', density=True)
    ax.hist(cos_neg, bins=bins, color=RED,   alpha=0.65,
            label=f'Negative edges  (n={len(cos_neg):,})', density=True)
    ax.axvline(0, color='#2c3e50', ls='--', lw=1.2,
               label='cos θ = 0  (90° boundary)')
    ax.axvline(np.mean(cos_pos), color=GREEN, ls=':', lw=1.5,
               label=f'Mean pos = {np.mean(cos_pos):.3f}')
    ax.axvline(np.mean(cos_neg), color=RED,   ls=':', lw=1.5,
               label=f'Mean neg = {np.mean(cos_neg):.3f}')
    ax.set_xlabel(r'Cosine similarity  $\cos\theta_{uv} = \hat{z}_u \cdot \hat{z}_v$')
    ax.set_ylabel('Density')
    ax.set_title('Angular Separation by Edge Sign\n'
                 r'$\cos\theta > 0$ → positive, $\cos\theta < 0$ → negative',
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, alpha=0.3)

    # ── (c) Polar angle θ vs positive-ratio ──────────────────────────────────
    ax = axes[2]
    theta_nodes = np.degrees(np.arctan2(emb_unit[:, 1], emb_unit[:, 0]))
    sc2 = ax.scatter(theta_nodes, pos_ratio,
                     c=log_deg, cmap=CMAP_DEG,
                     s=30, alpha=0.75, linewidths=0.3, edgecolors='#7f8c8d')
    plt.colorbar(sc2, ax=ax, label='log(1 + degree)', shrink=0.75, pad=0.02)
    # Annotate high-degree nodes
    for i in np.argsort(total_deg)[-min(15, args.label_top):]:
        ax.annotate(_short(idx_to_label[i]),
                    (theta_nodes[i], pos_ratio[i]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', color='#2c3e50')
    ax.axhline(0.5, color='#7f8c8d', ls='--', lw=1, label='Neutral ratio = 0.5')
    ax.set_xlabel('Polar angle θ = atan2(z₂, z₁)  [degrees]')
    ax.set_ylabel('Positive edge ratio')
    ax.set_title('Polar Angle vs Sentiment Profile\n'
                 'Nodes at similar θ share similar sentiment orientation',
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9); ax.grid(True, alpha=0.3)

else:
    # For proj_dim != 2: only the cosine similarity histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        'Angular Separation by Edge Sign\n'
        r'$\hat{y}_{uv} = \sigma(z_u \cdot z_v)$ — '
        r'$\cos\theta > 0 \Rightarrow$ positive, $\cos\theta < 0 \Rightarrow$ negative',
        fontsize=11, fontweight='bold')
    bins = np.linspace(-1, 1, 41)
    ax.hist(cos_pos, bins=bins, color=GREEN, alpha=0.65,
            label=f'Positive edges  (n={len(cos_pos):,})', density=True)
    ax.hist(cos_neg, bins=bins, color=RED,   alpha=0.65,
            label=f'Negative edges  (n={len(cos_neg):,})', density=True)
    ax.axvline(0, color='#2c3e50', ls='--', lw=1.2,
               label='cos θ = 0  (90° boundary)')
    ax.axvline(np.mean(cos_pos), color=GREEN, ls=':', lw=1.5,
               label=f'Mean pos = {np.mean(cos_pos):.3f}')
    ax.axvline(np.mean(cos_neg), color=RED,   ls=':', lw=1.5,
               label=f'Mean neg = {np.mean(cos_neg):.3f}')
    ax.set_xlabel(r'Cosine similarity  $\cos\theta_{uv} = \hat{z}_u \cdot \hat{z}_v$')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9, framealpha=0.9); ax.grid(True, alpha=0.3)

plt.tight_layout()
p = os.path.join(args.out_dir, 'proj_angular.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — proj_3d.png  (only when proj_dim ≥ 3)
# ─────────────────────────────────────────────────────────────────────────────
if args.proj_dim >= 3:
    fig  = plt.figure(figsize=(10, 8))
    ax3d = fig.add_subplot(111, projection='3d')
    sc   = ax3d.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                        c=pos_ratio, cmap=CMAP_SIGN, norm=norm_ratio,
                        s=25, alpha=0.75, depthshade=True)
    plt.colorbar(sc, ax=ax3d, label='Positive Edge Ratio', shrink=0.6, pad=0.1)
    for i in np.argsort(total_deg)[-min(15, args.label_top):]:
        ax3d.text(emb[i, 0], emb[i, 1], emb[i, 2],
                  _short(idx_to_label[i], 18), fontsize=6, color='#2c3e50')
    ax3d.set_title(f'3-D Projection — Positive Edge Ratio\n'
                   f'{N} nodes, {args.epochs} epochs', fontsize=11)
    ax3d.set_xlabel('Dim 1'); ax3d.set_ylabel('Dim 2'); ax3d.set_zlabel('Dim 3')
    plt.tight_layout()
    p = os.path.join(args.out_dir, 'proj_3d.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Nodes              : {N}")
print(f"Signed edges       : {len(df):,}  "
      f"(pos={df['sign'].eq(1).sum():,}, neg={df['sign'].eq(-1).sum():,})")
print(f"proj_dim           : {args.proj_dim}")
print(f"||WW^T - I||_F     : {ortho_err:.6f}")
print(f"Final task loss    : {task_losses[-1]:.4f}")
print(f"Final L_o          : {ortho_losses[-1]:.6f}")

print(f"\nTop-5 most positively perceived (min degree={args.min_degree}):")
for i in sort_ord[-5:][::-1]:
    print(f"  {active_ratios[i]:.3f}  {active_labels[i]}  (deg={active_deg[i]})")
print(f"\nTop-5 most negatively perceived:")
for i in sort_ord[:5]:
    print(f"  {active_ratios[i]:.3f}  {active_labels[i]}  (deg={active_deg[i]})")

out_files = ['proj_scatter.png', 'proj_extreme.png', 'proj_edges.png',
             'training_curves.png', 'sentiment_analysis.png',
             'projection_matrix.png', 'proj_angular.png']
if args.proj_dim >= 3:
    out_files.append('proj_3d.png')
if PLOTLY_AVAILABLE:
    out_files.append('proj_scatter_interactive.html')
if DASH_AVAILABLE:
    out_files.append('app_interactive.py (+ proj_data.pkl)')
print(f"\n✓  All figures saved to ./{args.out_dir}/")
print(f"   {' · '.join(out_files)}")
