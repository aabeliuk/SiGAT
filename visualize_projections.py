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

import difflib
import os
import random
import re
import unicodedata
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
    default='experiment-data/edgelist_fixed.csv')
parser.add_argument('--party_csv', default='experiment-data/congress_matched.csv',
    help='Optional node metadata CSV with party labels (node_name,node_id,partido)')
parser.add_argument('--out_dir',      default='visualizations')
parser.add_argument('--proj_dim',     type=int,   default=2,
    help='Dimension of the orthogonal projection (2 → 2-D, ≥3 → also 3-D)')
parser.add_argument('--embed_dim',    type=int,   default=20,
    help='Encoder embedding dimension (EMBEDDING_SIZE1 in sdgnn)')
parser.add_argument('--feat_dim',     type=int,   default=20,
    help='Node feature dimension (NODE_FEAT_SIZE in sdgnn)')
parser.add_argument('--epochs',       type=int,   default=30)
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
parser.add_argument('--min_degree',   type=int,   default=5,
    help='Minimum degree for nodes shown in analysis and filtering outputs')
parser.add_argument('--min_core_degree', type=int, default=5,
    help='Filter out nodes with degree <= this value before embedding')
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
VALID_SENTIMENTS = {'positive', 'negative', 'neutral'}

df[FROM_COL] = df[FROM_COL].astype(str).str.strip()
df[TO_COL]   = df[TO_COL].astype(str).str.strip()

# Drop malformed rows where the sentiment column contains description text
# instead of a valid sentiment label (column shift in the CSV).
n_before = len(df)
df = df[df[SIGN_COL].isin(VALID_SENTIMENTS)].copy()
n_dropped = n_before - len(df)
if n_dropped:
    print(f"  Dropped {n_dropped:,} malformed rows (sentiment column contained description text)")

df = df[df[FROM_COL] != df[TO_COL]]      # remove self-loops

# Separate neutral edges BEFORE dropping them — kept for graph structure only,
# not used in the sign-prediction loss.
df_neutral = df[df[SIGN_COL] == 'neutral'].copy()
df = df[df[SIGN_COL].isin(sign_map)].copy()
df['sign'] = df[SIGN_COL].map(sign_map).astype(int)
print(f"  After filtering: {len(df):,} signed edges "
      f"({df['sign'].eq(1).sum():,} pos, {df['sign'].eq(-1).sum():,} neg)"
      f"  +  {len(df_neutral):,} neutral edges (structure only)")

# Integer node mapping  (politician label → integer ID)
# Include nodes that appear only in neutral edges so they get embeddings.
all_labels   = sorted(set(df[FROM_COL]) | set(df[TO_COL])
                      | set(df_neutral[FROM_COL]) | set(df_neutral[TO_COL]))
N            = len(all_labels)
node_map     = {name: i for i, name in enumerate(all_labels)}
idx_to_label = {i: name for name, i in node_map.items()}
print(f"  {N} unique nodes")

def _normalize_text(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s or None


def _best_party_for_label(label, party_name_map):
    norm_label = _normalize_text(label)
    if not norm_label:
        return None
    # Skip obvious non-person strings: more than 5 tokens is almost certainly
    # an organization name, a quote, or a collective noun — not a politician.
    if len(norm_label.split()) > 5:
        return None
    # 1. Exact match
    if norm_label in party_name_map:
        return party_name_map[norm_label]
    # 2. Fuzzy character-level match (handles accents, typos)
    best = difflib.get_close_matches(norm_label, party_name_map.keys(), n=1,
                                     cutoff=0.78)
    if best:
        return party_name_map[best[0]]
    # 3. Substring containment — only if the label is short (≤ 3 tokens)
    #    to avoid "Gobierno de Gabriel Boric" → "gabriel boric" → Independiente
    if len(norm_label.split()) <= 3:
        for key in party_name_map:
            if key in norm_label or norm_label in key:
                return party_name_map[key]
    # 4. Token subset: all tokens of the shorter name appear in the longer name
    #    e.g. "jose kast" matches "jose antonio kast"
    label_tokens = set(norm_label.split())
    for key in party_name_map:
        key_tokens = set(key.split())
        shorter, longer = (
            (label_tokens, key_tokens) if len(label_tokens) <= len(key_tokens)
            else (key_tokens, label_tokens)
        )
        if len(shorter) >= 2 and shorter.issubset(longer):
            return party_name_map[key]
    return None


# Load party metadata for node colouring if available
party_name_map = {}
if args.party_csv:
    try:
        party_df = pd.read_csv(args.party_csv)
        # Prefer the 'bloc' column; fall back to 'partido' for compatibility
        party_df = party_df.rename(columns={
            'Bloc': 'bloc', 'bloc': 'bloc',
            'Partido': 'partido', 'partido': 'partido',
            'node_name': 'node_name', 'node_id': 'node_id',
            'cong_name': 'cong_name',
        })
        for _, row in party_df.iterrows():
            # prefer bloc, then partido
            partido = None
            if 'bloc' in row:
                partido = row.get('bloc')
            if (pd.isna(partido) or partido == '') and 'partido' in row:
                partido = row.get('partido')
            if pd.isna(partido) or partido == '':
                continue
            partido = str(partido).strip()
            if not partido:
                continue

            for col in ('cong_name', 'node_name'):
                name = _normalize_text(row.get(col))
                if name:
                    party_name_map[name] = partido
    except FileNotFoundError:
        print(f"Warning: party CSV not found: {args.party_csv}")
    except Exception as exc:
        print(f"Warning: failed to load party CSV {args.party_csv}: {exc}")

# Filter out nodes with degree <= min_core_degree before embedding.
# Keep only nodes that remain in the k-core of the signed graph.
keep_labels = set(all_labels)
while True:
    deg_counts = defaultdict(int)
    for _, row in df[df[FROM_COL].isin(keep_labels) &
                    df[TO_COL].isin(keep_labels)].iterrows():
        deg_counts[row[FROM_COL]] += 1
        deg_counts[row[TO_COL]] += 1

    to_remove = {label for label in keep_labels if deg_counts[label] <= args.min_core_degree}
    if not to_remove:
        break
    keep_labels -= to_remove

if len(keep_labels) != len(all_labels):
    print(f"Filtering {len(all_labels) - len(keep_labels)} nodes with degree <= {args.min_core_degree} before embedding")
    df = df[df[FROM_COL].isin(keep_labels) & df[TO_COL].isin(keep_labels)].copy()
    all_labels = sorted(keep_labels)
    N = len(all_labels)
    node_map = {name: i for i, name in enumerate(all_labels)}
    idx_to_label = {i: name for name, i in node_map.items()}
    print(f"  {N} unique nodes remain after filtering")

# Recompute party labels for the filtered node set.
party_labels = np.array(['Unknown'] * N, dtype=object)
if party_name_map:
    for i, label in idx_to_label.items():
        party = _best_party_for_label(label, party_name_map)
        if party:
            party_labels[i] = party

# Debug: show party label distribution
from collections import Counter
_label_counts = Counter(party_labels)
print("  Party label distribution (top 15):")
for _lbl, _cnt in _label_counts.most_common(15):
    print(f"    {_lbl!r}: {_cnt}")

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

# ── Step 1: accumulate per-sign counts for every directed (u, v) pair ────────
# pos_counts[u][v] = number of positive interactions u → v
# neg_counts[u][v] = number of negative interactions u → v
pos_counts = defaultdict(lambda: defaultdict(int))
neg_counts = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    u = node_map[row[FROM_COL]]
    v = node_map[row[TO_COL]]
    if int(row['sign']) == 1:
        pos_counts[u][v] += 1
    else:
        neg_counts[u][v] += 1

# ── Step 2: resolve each (u, v) to a single sign via weighted majority ────────
# All unique directed pairs that appear in at least one signed edge
all_pairs = set()
for u, neighbors in pos_counts.items():
    for v in neighbors: all_pairs.add((u, v))
for u, neighbors in neg_counts.items():
    for v in neighbors: all_pairs.add((u, v))

# Total interaction count per pair (pos + neg) — used for log-normalised weight
edge_counts = defaultdict(lambda: defaultdict(int))
for u, v in all_pairs:
    edge_counts[u][v] = pos_counts[u][v] + neg_counts[u][v]

# Log-normalised weight: w(u,v) = log(1 + total_count) / log(1 + max_count)
_max_count = max(
    (c for neighbors in edge_counts.values() for c in neighbors.values()),
    default=1
)
_log_max = np.log1p(_max_count)

def _lognorm(count):
    return np.log1p(count) / _log_max

# Weighted majority: positive wins if Σw(pos edges) > Σw(neg edges)
# i.e. log(1+pos_count) > log(1+neg_count)  ⟺  pos_count > neg_count
# (log is monotone, so this simplifies to raw count comparison — but we keep
#  the log form for clarity and future flexibility)
n_resolved_pos = n_resolved_neg = n_ties = 0
resolved_sign = {}   # (u, v) → +1 or -1
for u, v in all_pairs:
    w_pos = np.log1p(pos_counts[u][v])
    w_neg = np.log1p(neg_counts[u][v])
    if w_pos >= w_neg:          # ties go to positive (more charitable)
        resolved_sign[(u, v)] = 1
        n_resolved_pos += 1
    else:
        resolved_sign[(u, v)] = -1
        n_resolved_neg += 1

n_ties = sum(1 for (u,v) in all_pairs
             if pos_counts[u][v] == neg_counts[u][v] and pos_counts[u][v] > 0)
print(f"  Weighted-majority sign resolution: "
      f"{n_resolved_pos:,} positive, {n_resolved_neg:,} negative  "
      f"({n_ties} ties → positive)")

# ── Step 3: build adjacency lists from resolved signs ────────────────────────
adj_lists1   = defaultdict(set)
adj_lists1_1 = defaultdict(set)
adj_lists1_2 = defaultdict(set)
adj_lists2   = defaultdict(set)
adj_lists2_1 = defaultdict(set)
adj_lists2_2 = defaultdict(set)
# Neutral edges: two directed channels (out / in) used only by the encoder,
# never by the sign-prediction loss.
adj_lists3_1 = defaultdict(set)   # neutral  u → v
adj_lists3_2 = defaultdict(set)   # neutral  v ← u

for (u, v), sign in resolved_sign.items():
    if sign == 1:
        adj_lists1[u].add(v);   adj_lists1[v].add(u)
        adj_lists1_1[u].add(v); adj_lists1_2[v].add(u)
    else:
        adj_lists2[u].add(v);   adj_lists2[v].add(u)
        adj_lists2_1[u].add(v); adj_lists2_2[v].add(u)

# Add neutral edges to the graph structure (structure only — not in loss)
for _, row in df_neutral.iterrows():
    u_lbl, v_lbl = row[FROM_COL], row[TO_COL]
    if u_lbl not in node_map or v_lbl not in node_map:
        continue
    u = node_map[u_lbl]
    v = node_map[v_lbl]
    adj_lists3_1[u].add(v)
    adj_lists3_2[v].add(u)

n_neutral_edges = sum(len(vs) for vs in adj_lists3_1.values())
print(f"  Neutral edges added to encoder graph: {n_neutral_edges:,} directed arcs")

# ── Step 4: log-normalised weight dict ───────────────────────────────────────
_min_weight = _lognorm(1)
weight_dict = defaultdict(dict)
for u, neighbors in edge_counts.items():
    for v, count in neighbors.items():
        weight_dict[u][v] = _lognorm(count)

print(f"  Edge weights: log-normalised, max_count={_max_count}, "
      f"range [{_min_weight:.3f}, 1.000]")

# Convert directed adj_lists → scipy sparse matrices for sdgnn.Encoder
# Same transformation as sdgnn.run()'s inner `func` helper.
def _to_csr(adj_dict: dict, n: int) -> sp.csr_matrix:
    edges = [(u, v) for u, vs in adj_dict.items() for v in vs]
    if not edges:
        return sp.csr_matrix((n, n))
    rows, cols = zip(*edges)
    return sp.csr_matrix(
        (np.ones(len(edges)), (rows, cols)), shape=(n, n))

# 6 adjacency channels: pos-out, pos-in, neg-out, neg-in, neutral-out, neutral-in
adj_sparse = [_to_csr(d, N)
              for d in (adj_lists1_1, adj_lists1_2,
                        adj_lists2_1, adj_lists2_2,
                        adj_lists3_1, adj_lists3_2)]

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
LEFT_ANCHOR_NAMES = [
    'Gabriel Boric', 'Camila Vallejo', 'Karol Cariola',
    'Giorgio Jackson', 'Lautaro Carmona'
]
RIGHT_ANCHOR_NAMES = [
    'José Antonio Kast', 'Juan Antonio Coloma', 'Evelyn Matthei',
    'Sebastián Piñera', 'Iván Moreira'
]


def _find_indices_by_names(names):
    wanted = {_normalize_text(name): name for name in names if _normalize_text(name)}
    return [i for i, label in idx_to_label.items()
            if _normalize_text(label) in wanted]


def _anchor_projection(raw_emb, left_idx, right_idx):
    """Return a 1-D projection that best separates left vs right anchors.

    The axis is the (left_centroid - right_centroid) direction in the raw
    embedding space, applied to mean-centered embeddings. Returns a
    1-D numpy array of length N, or None on failure.
    """
    if len(left_idx) == 0 or len(right_idx) == 0:
        return None
    centered = raw_emb - raw_emb.mean(axis=0)
    left_centroid = centered[left_idx].mean(axis=0)
    right_centroid = centered[right_idx].mean(axis=0)
    direction = left_centroid - right_centroid
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None
    basis = direction / norm
    return centered @ basis

model.eval()
with torch.no_grad():
    emb = model.forward(list(range(N))).cpu().numpy()   # (N, proj_dim)
    raw_emb = model.enc(list(range(N))).cpu().numpy()   # (N, embed_dim)

W_np      = model.proj.projection.weight.detach().cpu().numpy()   # (proj_dim, embed_dim)
WWT       = W_np @ W_np.T
I_        = np.eye(W_np.shape[0])
ortho_err = float(np.linalg.norm(WWT - I_, 'fro'))
print(f"\nEmbeddings : {emb.shape}  |  ||WW^T - I||_F = {ortho_err:.6f}")

left_anchor_idx = _find_indices_by_names(LEFT_ANCHOR_NAMES)
right_anchor_idx = _find_indices_by_names(RIGHT_ANCHOR_NAMES)
anchor_emb = None
if len(left_anchor_idx) > 0 and len(right_anchor_idx) > 0:
    anchor_emb = _anchor_projection(raw_emb, left_anchor_idx, right_anchor_idx)
    print(f"Anchor projection: {len(left_anchor_idx)} left anchors, {len(right_anchor_idx)} right anchors")
else:
    print("Anchor projection skipped: missing anchor labels")

# ─────────────────────────────────────────────────────────────────────────────
# LDA ideology probe
# ─────────────────────────────────────────────────────────────────────────────
# Assign a binary left/right label to every node that has a known bloc.
# Left bloc:  PC, Frente Amplio, PS, PPD, PRSD, PH
# Right bloc: UDI, RN, Republicano, Evopoli
# LEFT_BLOCS  = {'pc', 'frente amplio', 'ps', 'ppd', 'prsd', 'ph'}
# RIGHT_BLOCS = {'udi', 'rn', 'republicano', 'evopoli'}
LEFT_BLOCS  = {'pc', 'ph', 'frente amplio'}
RIGHT_BLOCS = {'udi', 'republicano'}

lda_ideology   = None   # (N,) scores for all nodes — filled if LDA succeeds
lda_coef       = None   # (embed_dim,) weight vector
lda_score      = None   # held-out accuracy

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Build labelled set
    lda_indices, lda_labels = [], []
    for i, lbl in idx_to_label.items():
        bloc_norm = _normalize_text(party_labels[i])
        if bloc_norm in LEFT_BLOCS:
            lda_indices.append(i)
            lda_labels.append(0)   # 0 = left
        elif bloc_norm in RIGHT_BLOCS:
            lda_indices.append(i)
            lda_labels.append(1)   # 1 = right

    lda_indices = np.array(lda_indices)
    lda_labels  = np.array(lda_labels)
    n_left  = (lda_labels == 0).sum()
    n_right = (lda_labels == 1).sum()
    print(f"\nLDA ideology probe: {n_left} left nodes, {n_right} right nodes")

    if min(n_left, n_right) >= 5:
        X_lda = raw_emb[lda_indices]          # (n_labelled, embed_dim)
        lda   = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X_lda, lda_labels)

        # Score via 5-fold CV
        cv_scores  = cross_val_score(lda, X_lda, lda_labels, cv=min(5, min(n_left, n_right)), scoring='accuracy')
        lda_score  = cv_scores.mean()

        # Fit on full labelled set to get direction + scores for all nodes
        lda_ideology = lda.transform(raw_emb).squeeze()   # (N,)
        lda_coef     = lda.coef_.squeeze()                 # (embed_dim,)
        print(f"LDA 5-fold CV accuracy: {lda_score:.3f}  |  ideology axis dim: {lda_coef.shape}")
    else:
        print("LDA skipped: not enough labelled nodes per side (need ≥ 5 each)")
except ImportError:
    print("LDA skipped: scikit-learn not available (pip install scikit-learn)")

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


def _clean_display_name(s: str) -> str:
    """Return a human-friendly, trimmed display name for party labels.

    Keeps original accents and capitalization where possible but collapses
    weird whitespace and non-printable characters.
    """
    if s is None:
        return 'Unknown'
    t = str(s).strip()
    # remove control / non-printable chars
    t = ''.join(ch for ch in t if ch.isprintable())
    # collapse inner whitespace
    t = re.sub(r'\s+', ' ', t)
    return t



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
num_panels = 4 if anchor_emb is not None else 3
fig_w = 24 if num_panels == 4 else 20
fig, axes = plt.subplots(1, num_panels, figsize=(fig_w, 6.5))
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

if anchor_emb is not None:
    ax_anchor = axes[3]
    # 1-D projection: plot along x with a small reproducible jitter on y
    rng = np.random.RandomState(args.seed)
    jitter = rng.normal(scale=0.02, size=anchor_emb.shape[0])
    sc_anchor = ax_anchor.scatter(anchor_emb, jitter,
                    c=pos_ratio, cmap=CMAP_SIGN, norm=norm_ratio,
                    s=35, alpha=0.85, linewidths=0.3, edgecolors='#7f8c8d')
    plt.colorbar(sc_anchor, ax=ax_anchor, label='Positive Edge Ratio', shrink=0.8, pad=0.01)
    ax_anchor.set_title('Anchor-based 1-D Projection (Left vs Right)', fontsize=10)
    ax_anchor.set_xlabel('Anchor axis (left ← right)')
    ax_anchor.get_yaxis().set_visible(False)
    # annotate anchor nodes at their x positions
    for i in left_anchor_idx:
        ax_anchor.annotate(_short(idx_to_label[i], maxlen=18),
                    (anchor_emb[i], 0.03), fontsize=8, ha='center', va='bottom',
                    color='#1f618d')
    for i in right_anchor_idx:
        ax_anchor.annotate(_short(idx_to_label[i], maxlen=18),
                    (anchor_emb[i], 0.03), fontsize=8, ha='center', va='bottom',
                    color='#922b21')

plt.tight_layout()
p = os.path.join(args.out_dir, 'proj_scatter.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  ✓  {p}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1a — proj_party.png  (2-D scatter by party membership)
party_position_map = {
    # Far left
    'ph': -1.0,
    'pc': -0.85,
    'partido comunista de chile': -0.85,
    # Left
    'frente amplio': -0.75,
    'revolucion democratica': -0.7,
    'convergencia social': -0.7,
    'ps': -0.6,
    'partido socialista de chile': -0.6,
    'ppd': -0.45,
    'partido por la democracia': -0.45,
    'izquierda ciudadana': -0.5,
    'partido ecologista verde': -0.4,
    'federacion regionalista verde social': -0.35,
    # Centre-left
    'prsd': -0.2,
    'partido radical socialdemocrata': -0.2,
    'partido liberal de chile': -0.15,
    # Centre
    'dc': 0.0,
    'partido democrata cristiano': 0.0,
    'independiente': 0.0,
    'independientes': 0.0,
    # Centre-right
    'democratas': 0.2,
    'partido de la gente': 0.25,
    'partido social cristiano': 0.3,
    'amplitud': 0.35,
    'evopoli': 0.5,
    'partido evolucion politica': 0.5,
    # Right
    'rn': 0.7,
    'partido renovacion nacional': 0.7,
    'udi': 0.85,
    'partido union democrata independiente': 0.85,
    # Far right
    'republicano': 1.0,
    'partido republicano': 1.0,
    'partido nacional libertario': 0.95,
}

unique_parties = sorted(set(party_labels))
if len(unique_parties) > 1 and any(p != 'Unknown' for p in unique_parties):
    # Exclude 'Unknown' entirely from the party plot
    known_party_order = [
        'PH', 'PC', 'Frente Amplio', 'PS', 'PPD', 'PRSD', 'DC',
        'Independientes', 'Demócratas', 'Evópoli', 'RN', 'UDI', 'Republicano'
    ]
    parties = []
    for canonical in known_party_order:
        match = next((p for p in unique_parties if _normalize_text(p) == _normalize_text(canonical)), None)
        if match:
            parties.append(match)
    parties.extend([p for p in unique_parties
                    if p != 'Unknown'
                    and p not in parties
                    and _normalize_text(p) in party_position_map])

    norm = Normalize(vmin=-1.0, vmax=1.0)
    cmap = plt.get_cmap('RdYlGn')
    fig, ax = plt.subplots(figsize=(12, 10))
    for party in parties:
        mask = party_labels == party
        if not np.any(mask):
            continue
        position = party_position_map.get(_normalize_text(party), 0.0)
        display_name = _clean_display_name(party)
        short_name = _short(display_name, maxlen=18)
        ax.scatter(emb[mask, 0], emb[mask, 1], c=np.full(mask.sum(), position), cmap=cmap,
                   norm=norm, s=45, alpha=0.85, linewidths=0.4,
                   edgecolors='none', label=f"{short_name} ({mask.sum()})")

    # Annotate specific well-known left/right politicians for clarity
    left_names = [
        'Gabriel Boric', 'Camila Vallejo', 'Karol Cariola',
        'Giorgio Jackson', 'Lautaro Carmona'
    ]
    right_names = [
        'José Antonio Kast', 'Juan Antonio Coloma', 'Evelyn Matthei',
        'Sebastián Piñera', 'Iván Moreira'
    ]

    def _find_indices_by_names(names):
        wanted = {(_normalize_text(n) or ''): n for n in names}
        found = []
        for i, lab in idx_to_label.items():
            if lab is None:
                continue
            if _normalize_text(lab) in wanted:
                found.append(i)
        return found

    left_idx = _find_indices_by_names(left_names)
    right_idx = _find_indices_by_names(right_names)

    for i in left_idx:
        ax.annotate(_short(idx_to_label[i], maxlen=20), (emb[i, 0], emb[i, 1]),
                    fontsize=8, ha='center', va='bottom', xytext=(0, 6),
                    textcoords='offset points', color='#1f618d',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                              ec='#1f618d', lw=0.6))

    for i in right_idx:
        ax.annotate(_short(idx_to_label[i], maxlen=20), (emb[i, 0], emb[i, 1]),
                    fontsize=8, ha='center', va='bottom', xytext=(0, 6),
                    textcoords='offset points', color='#922b21',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                              ec='#922b21', lw=0.6))
    ax.set_title('2-D Projection Colored by Party Position', fontsize=12)
    ax.set_xlabel('Proj. Dim 1'); ax.set_ylabel('Proj. Dim 2')
    # Place legend outside plot area so it isn't clipped when saving
    # use multiple legend columns for many parties to avoid long vertical stacks
    n_legend_cols = 1 if len(parties) <= 8 else (2 if len(parties) <= 16 else 3)
    ax.legend(fontsize=9, framealpha=0.9, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=n_legend_cols,
              handlelength=1.5)
    ax.grid(True, alpha=0.25)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Party Position', rotation=270, labelpad=15)
    plt.tight_layout()
    p_party = os.path.join(args.out_dir, 'proj_party.png')
    plt.savefig(p_party, dpi=150, bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f"  ✓  {p_party}")

    if anchor_emb is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        rng = np.random.RandomState(args.seed + 1)
        jitter = rng.normal(scale=0.01, size=anchor_emb.shape[0])
        for party in parties:
            mask = party_labels == party
            if not np.any(mask):
                continue
            position = party_position_map.get(_normalize_text(party), 0.0)
            ax.scatter(anchor_emb[mask], jitter[mask], c=np.full(mask.sum(), position), cmap=cmap,
                       norm=norm, s=45, alpha=0.85, linewidths=0.4,
                       edgecolors='none', label=f"{_short(_clean_display_name(party), maxlen=18)} ({mask.sum()})")

        for i in left_idx:
            ax.annotate(_short(idx_to_label[i], maxlen=20), (anchor_emb[i], jitter[i] + 0.04),
                        fontsize=8, ha='center', va='bottom', xytext=(0, 6),
                        textcoords='offset points', color='#1f618d',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                                  ec='#1f618d', lw=0.6))
        for i in right_idx:
            ax.annotate(_short(idx_to_label[i], maxlen=20), (anchor_emb[i], jitter[i] + 0.04),
                        fontsize=8, ha='center', va='bottom', xytext=(0, 6),
                        textcoords='offset points', color='#922b21',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                                  ec='#922b21', lw=0.6))

        ax.set_title('1-D Anchor Projection Colored by Party Position', fontsize=12)
        ax.set_xlabel('Anchor axis (left ← right)')
        ax.get_yaxis().set_visible(False)
        ax.grid(True, axis='x', alpha=0.25)
        n_legend_cols = 1 if len(parties) <= 8 else (2 if len(parties) <= 16 else 3)
        ax.legend(fontsize=9, framealpha=0.9, loc='upper left',
                  bbox_to_anchor=(1.02, 1), borderaxespad=0., ncol=n_legend_cols,
                  handlelength=1.5)
        plt.tight_layout()
        p_party_1d = os.path.join(args.out_dir, 'proj_party_1d.png')
        plt.savefig(p_party_1d, dpi=150, bbox_inches='tight', pad_inches=0.08)
        plt.close()
        print(f"  ✓  {p_party_1d}")
else:
    print("Skipping party-coloured plot: no valid party labels found.")

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
    # Exclude nodes without a matched bloc, or whose bloc has no position entry.
    mask = np.array([
        lbl != 'Unknown' and _normalize_text(lbl) in party_position_map
        for lbl in party_labels
    ])
    if not np.any(mask):
        print('Skipping interactive scatter: no matched bloc labels')
    else:
        emb_m = emb[mask]
        total_deg_m = total_deg[mask]
        # idx_to_label may be a list; collect matching labels
        idxs = [i for i, m in enumerate(mask) if m]
        labels_m = [idx_to_label[i] for i in idxs]
        blocs_m = party_labels[mask]
        bloc_positions = np.array([
            party_position_map.get(_normalize_text(bloc), 0.0)
            for bloc in blocs_m
        ])

        # Build per-node edge coordinate maps for interactive hover-rendering.
        # Only include edges that connect visible nodes in the filtered scatter.
        visible_nodes = set(idxs)
        pos_edge_coords = {}
        neg_edge_coords = {}
        # idxs maps positions in emb_m -> original node index
        for local_i, orig_idx in enumerate(idxs):
            px, py = float(emb[orig_idx, 0]), float(emb[orig_idx, 1])
            # positive outgoing
            xs_p, ys_p = [], []
            for v in adj_lists1_1.get(orig_idx, []):
                if v not in visible_nodes:
                    continue
                xs_p.extend([px, float(emb[v, 0]), None])
                ys_p.extend([py, float(emb[v, 1]), None])
            # negative outgoing
            xs_n, ys_n = [], []
            for v in adj_lists2_1.get(orig_idx, []):
                if v not in visible_nodes:
                    continue
                xs_n.extend([px, float(emb[v, 0]), None])
                ys_n.extend([py, float(emb[v, 1]), None])
            pos_edge_coords[str(orig_idx)] = {'x': xs_p, 'y': ys_p}
            neg_edge_coords[str(orig_idx)] = {'x': xs_n, 'y': ys_n}

        # Main node scatter plus two empty line traces for edges (pos / neg)
        scatter_html = go.Figure(
            data=[
                go.Scatter(
                    x=emb_m[:, 0],
                    y=emb_m[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=bloc_positions,
                        colorscale='RdYlGn',
                        colorbar=dict(title='Bloc Position'),
                        cmin=-1.0,
                        cmax=1.0,
                        line=dict(width=0.3, color='#7f8c8d'),
                    ),
                    text=[
                        f"<b>{labels_m[i]}</b><br>bloc={blocs_m[i]}<br>"
                        f"degree={int(total_deg_m[i])}<br>"
                        f"position={bloc_positions[i]:.2f}"
                        for i in range(len(idxs))
                    ],
                    hoverinfo='text',
                    customdata=idxs,
                ),
                go.Scatter(
                    x=[], y=[], mode='lines', hoverinfo='none',
                    line=dict(color='rgba(39,174,96,0.9)', width=2),
                    name='positive edges', showlegend=False
                ),
                go.Scatter(
                    x=[], y=[], mode='lines', hoverinfo='none',
                    line=dict(color='rgba(192,57,43,0.9)', width=2, dash='dash'),
                    name='negative edges', showlegend=False
                ),
            ],
            layout=go.Layout(
                title='Interactive 2-D Projection — Bloc Position',
                xaxis=dict(title='Proj. Dim 1'),
                yaxis=dict(title='Proj. Dim 2'),
                hovermode='closest',
            )
        )

        # Write HTML with embedded JS that listens for hover and updates edge traces
        html_path = os.path.join(args.out_dir, 'proj_scatter_interactive.html')
        import json
        pos_json = json.dumps(pos_edge_coords)
        neg_json = json.dumps(neg_edge_coords)

        html_str = pio.to_html(scatter_html, full_html=True, include_plotlyjs='cdn', div_id='proj-scatter')
        script = f"""
<script>
const POS_EDGES = {pos_json};
const NEG_EDGES = {neg_json};
const gd = document.getElementById('proj-scatter');
if (gd) {{
  gd.on('plotly_hover', function(data) {{
    const pt = data.points[0];
    const orig = String(pt.customdata);
    const pos = POS_EDGES[orig] || {{x:[], y:[]}};
    const neg = NEG_EDGES[orig] || {{x:[], y:[]}};
    // traces: 0 = nodes, 1 = pos edges, 2 = neg edges
    Plotly.restyle(gd, {{x: [pos.x], y: [pos.y]}}, [1]);
    Plotly.restyle(gd, {{x: [neg.x], y: [neg.y]}}, [2]);
  }});
  gd.on('plotly_unhover', function() {{
    Plotly.restyle(gd, {{x: [[]], y: [[]]}}, [1]);
    Plotly.restyle(gd, {{x: [[]], y: [[]]}}, [2]);
  }});
}}
</script>
"""
        with open(html_path, 'w', encoding='utf-8') as fh:
            fh.write(html_str)
            fh.write(script)
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
# Figure 8 — lda_ideology.png
#
# Three panels:
#   (a) 1-D strip plot of LDA ideology scores, nodes coloured by party
#   (b) Per-dimension loadings: which of the embed_dim raw dimensions drive
#       the ideology axis (bar chart of lda_coef)
#   (c) 2-D scatter of the two highest-loading raw dimensions, coloured by
#       party — useful sanity check that the axis isn't degenerate
# ─────────────────────────────────────────────────────────────────────────────
if lda_ideology is not None and len(lda_indices) > 0:
    _lda_norm   = Normalize(vmin=-1.0, vmax=1.0)
    _lda_cmap   = plt.get_cmap('RdYlGn')

    # Map each labelled node to a numeric party position for colouring
    _bloc_pos = {
        'ph': -1.0, 'pc': -0.85, 'frente amplio': -0.8,
        'ps': -0.55, 'ppd': -0.4, 'prsd': -0.2,
        'dc': 0.0,
        'evopoli': 0.5, 'rn': 0.7, 'udi': 0.85, 'republicano': 1.0,
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        'LDA Ideology Probe on Raw Embeddings\n'
        f'embed_dim={args.embed_dim}  |  5-fold CV accuracy = {lda_score:.3f}'
        f'  |  {n_left} left, {n_right} right nodes labelled',
        fontsize=12, fontweight='bold')

    # ── (a) 1-D ideology strip ──────────────────────────────────────────────
    ax = axes[0]
    rng_lda = np.random.RandomState(args.seed + 42)

    # Pre-compute one jitter value per node so scatter points and annotation
    # arrows both land at exactly the same y-coordinate.
    node_jitter = rng_lda.normal(scale=0.08, size=N)

    # Plot ALL nodes with known blocs (not just left/right training set)
    known_bloc_parties = sorted(
        set(_normalize_text(party_labels[i])
            for i in range(N) if party_labels[i] != 'Unknown'),
        key=lambda b: _bloc_pos.get(b, 0.0)
    )
    for bloc_norm in known_bloc_parties:
        mask_b = np.array([_normalize_text(party_labels[i]) == bloc_norm for i in range(N)])
        if not np.any(mask_b):
            continue
        pos = _bloc_pos.get(bloc_norm, 0.0)
        raw_name = next((party_labels[i] for i in range(N) if _normalize_text(party_labels[i]) == bloc_norm), bloc_norm)
        display  = _clean_display_name(raw_name)
        ax.scatter(lda_ideology[mask_b], node_jitter[mask_b],
                   c=np.full(mask_b.sum(), pos), cmap=_lda_cmap, norm=_lda_norm,
                   s=40, alpha=0.85, linewidths=0.3, edgecolors='#7f8c8d',
                   label=f"{_short(display, 16)} ({mask_b.sum()})")

    # Annotate anchor politicians — reuse the same node_jitter[i] as the dot
    _ann_names = LEFT_ANCHOR_NAMES + RIGHT_ANCHOR_NAMES
    for i, lbl in idx_to_label.items():
        if any(_normalize_text(lbl) == _normalize_text(n) for n in _ann_names):
            is_left_ann = any(_normalize_text(lbl) == _normalize_text(n) for n in LEFT_ANCHOR_NAMES)
            col = '#1f618d' if is_left_ann else '#922b21'
            ax.annotate(
                _short(lbl, 20),
                xy=(lda_ideology[i], node_jitter[i]),          # exact dot position
                xytext=(0, 10), textcoords='offset points',    # label 10 pt above
                fontsize=7, ha='center', color=col,
                arrowprops=dict(arrowstyle='-', color=col, lw=0.6),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85,
                          ec=col, lw=0.6))

    ax.axvline(0, color='#7f8c8d', ls='--', lw=1.2, alpha=0.7, label='Decision boundary')
    ax.set_xlabel('LDA ideology score  (left ← 0 → right)')
    ax.get_yaxis().set_visible(False)
    ax.set_title('1-D LDA Ideology Axis\n(all nodes with known bloc, coloured by party)')
    ax.grid(True, axis='x', alpha=0.25)
    n_cols = 1 if len(known_bloc_parties) <= 8 else 2
    ax.legend(fontsize=8, framealpha=0.9, loc='upper left',
              bbox_to_anchor=(0.0, -0.08), borderaxespad=0., ncol=n_cols)

    # ── (b) embedding dimension loadings ────────────────────────────────────
    ax = axes[1]
    dims = np.arange(len(lda_coef))
    colors_bar = ['#c0392b' if c > 0 else '#2980b9' for c in lda_coef]
    ax.bar(dims, lda_coef, color=colors_bar, alpha=0.85, edgecolor='white', lw=0.4)
    # Mark top-5 most important dimensions
    top5 = np.argsort(np.abs(lda_coef))[-5:]
    for d in top5:
        ax.bar(d, lda_coef[d],
               color='#e67e22' if lda_coef[d] > 0 else '#8e44ad',
               alpha=0.95, edgecolor='white', lw=0.4,
               label=f'dim {d} ({lda_coef[d]:+.3f})' if d == top5[-1] else None)
        ax.annotate(f'd{d}', (d, lda_coef[d] + np.sign(lda_coef[d]) * 0.005),
                    ha='center', va='bottom' if lda_coef[d] >= 0 else 'top',
                    fontsize=7, color='#2c3e50')
    ax.axhline(0, color='#7f8c8d', lw=0.8)
    ax.set_xlabel('Raw Embedding Dimension')
    ax.set_ylabel('LDA Coefficient')
    ax.set_title('Ideology Axis Loadings\n'
                 'Which raw dims drive left vs right?\n'
                 '(orange/purple = top-5 by |coef|)')
    ax.set_xticks(dims[::2])
    ax.grid(True, axis='y', alpha=0.3)

    # ── (c) 2-D scatter of the two top-loading dims ──────────────────────────
    ax = axes[2]
    top2 = np.argsort(np.abs(lda_coef))[-2:][::-1]
    d0, d1 = int(top2[0]), int(top2[1])
    for bloc_norm in known_bloc_parties:
        mask_b = np.array([_normalize_text(party_labels[i]) == bloc_norm for i in range(N)])
        if not np.any(mask_b):
            continue
        pos = _bloc_pos.get(bloc_norm, 0.0)
        raw_name = next((party_labels[i] for i in range(N) if _normalize_text(party_labels[i]) == bloc_norm), bloc_norm)
        display  = _clean_display_name(raw_name)
        ax.scatter(raw_emb[mask_b, d0], raw_emb[mask_b, d1],
                   c=np.full(mask_b.sum(), pos), cmap=_lda_cmap, norm=_lda_norm,
                   s=40, alpha=0.85, linewidths=0.3, edgecolors='#7f8c8d',
                   label=_short(display, 16))
    ax.set_xlabel(f'Raw embedding dim {d0}  (coef={lda_coef[d0]:+.3f})')
    ax.set_ylabel(f'Raw embedding dim {d1}  (coef={lda_coef[d1]:+.3f})')
    ax.set_title(f'Top-2 ideology-loading raw dims\n(dim {d0} vs dim {d1})')
    ax.grid(True, alpha=0.25)
    sm_lda = plt.cm.ScalarMappable(norm=_lda_norm, cmap=_lda_cmap)
    sm_lda.set_array([])
    fig.colorbar(sm_lda, ax=ax, pad=0.02, label='Party position (left–right)')

    plt.tight_layout()
    p_lda = os.path.join(args.out_dir, 'lda_ideology.png')
    plt.savefig(p_lda, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  ✓  {p_lda}")
else:
    print("Skipping LDA ideology figure: probe did not run successfully.")

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
if lda_score is not None:
    print(f"LDA ideology CV acc: {lda_score:.3f}  ({n_left} left, {n_right} right nodes)")
    top5_dims = np.argsort(np.abs(lda_coef))[-5:][::-1]
    print("Top-5 ideology dims:", ", ".join(f"dim{d}({lda_coef[d]:+.3f})" for d in top5_dims))

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
