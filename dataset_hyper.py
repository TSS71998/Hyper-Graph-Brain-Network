import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from itertools import combinations

# ─────────────────────────────────────────────────────────────────
# HOC Co-Fluctuation & RSS Binning Utilities
# ─────────────────────────────────────────────────────────────────

def zscore(ts_raw, eps=1e-6):
    mu = ts_raw.mean(axis=0, keepdims=True)
    sig = ts_raw.std(axis=0, keepdims=True)
    return (ts_raw - mu) / (sig + eps)

def compute_rss(signal_ts):
    """signal_ts: (M, T) -> RSS (T,)"""
    return np.sqrt(np.sum(signal_ts ** 2, axis=0))

def rss_bin_means(signal_ts, K=5):
    """
    Sort T timepoints by overall RSS, partition into K activity bins,
    and compute average co-fluctuation per bin. Returns: (M, K)
    """
    M, T = signal_ts.shape
    rss = compute_rss(signal_ts)
    sort_idx = np.argsort(rss)
    bsz = T // K
    bins = np.zeros((M, K), dtype=np.float32)
    for k in range(K):
        bin_t = sort_idx[k * bsz : (k + 1) * bsz]
        bins[:, k] = signal_ts[:, bin_t].mean(axis=1)
    return bins

def compute_windowed_var(ts_signals, window_size=20, stride=5):
    """
    Computes sliding-window variance for signals of shape (M, T).
    Returns: (M, 1)
    """
    M, T = ts_signals.shape
    windows = []
    for start in range(0, T - window_size + 1, stride):
        w = ts_signals[:, start : start + window_size].mean(axis=1)
        windows.append(w)
    if len(windows) == 0:
        return np.zeros((M, 1), dtype=np.float32)
    return np.stack(windows, axis=1).var(axis=1, keepdims=True).astype(np.float32)

# ─────────────────────────────────────────────────────────────────
# Hypergraph Builder
# ─────────────────────────────────────────────────────────────────

def build_hypergraph(ts, k_neighbors=8, max_triangles_per_roi=3, K_bins=5):
    """
    Constructs an incident hypergraph with:
      1. k-NN cluster hyperedges around each node
      2. High-affinity triangle hyperedges (3-region dynamics)
    Returns:
      node_features: (N, N)
      hyperedge_index: (2, num_incidences) where row 0 = node, row 1 = hyperedge
      hyperedge_attr: (num_hyperedges, 1 + K_bins) [variance + K bins]
    """
    T, N = ts.shape
    ts_z = zscore(ts)  # (T, N)
    fc = np.corrcoef(ts_z.T)
    fc = np.nan_to_num(fc)

    hyperedges = []       # list of list of node indices
    hyperedge_types = []  # track grouping type

    # 1. k-NN Neighborhood Hyperedges (N hyperedges)
    for i in range(N):
        corr_row = np.abs(fc[i])
        corr_row[i] = -1.0
        top_k = np.argsort(corr_row)[-k_neighbors:].tolist()
        group = list(set([i] + top_k))
        hyperedges.append(group)
        hyperedge_types.append('knn')

    # 2. Triangle Hyperedges (Strongest 3-node cliques per region)
    for i in range(N):
        corr_row = np.abs(fc[i])
        corr_row[i] = -1.0
        top_cands = np.argsort(corr_row)[-5:].tolist()
        # Find strongest internal pairs among candidates
        tri_pairs = []
        for u, v in combinations(top_cands, 2):
            score = np.abs(fc[u, v]) + np.abs(fc[i, u]) + np.abs(fc[i, v])
            tri_pairs.append(((i, u, v), score))
        tri_pairs.sort(key=lambda x: x[1], reverse=True)
        for tri, _ in tri_pairs[:max_triangles_per_roi]:
            if sorted(tri) not in [sorted(h) for h in hyperedges]:
                hyperedges.append(list(tri))
                hyperedge_types.append('triangle')

    num_hyperedges = len(hyperedges)

    # ─────────────────────────────────────────────────────────────
    # Compute Hyperedge Time Series (hTS) and Features
    # ─────────────────────────────────────────────────────────────
    # For hyperedge e: hTS_e(t) = prod_{v in e} z_v(t)
    hTS = np.zeros((num_hyperedges, T), dtype=np.float32)
    for e_idx, members in enumerate(hyperedges):
        sub_signals = ts_z[:, members]  # (T, |e|)
        hTS[e_idx] = np.prod(sub_signals, axis=1)

    hDYN = compute_windowed_var(hTS, window_size=20, stride=5) # (E, 1)

    hCoF = rss_bin_means(hTS, K=K_bins)                          # (E, K)

    hyperedge_attr = np.concatenate([hDYN, hCoF], axis=1)      # (E, 1 + K)

    # Build 2-row incidence list: (node_idx, hyperedge_idx)
    inc_node = []
    inc_edge = []
    for e_idx, members in enumerate(hyperedges):
        for v in members:
            inc_node.append(v)
            inc_edge.append(e_idx)

    hyperedge_index = torch.tensor([inc_node, inc_edge], dtype=torch.long)
    hyperedge_attr = torch.tensor(hyperedge_attr, dtype=torch.float32)
    node_features = torch.tensor(fc, dtype=torch.float32)

    return hyperedge_index, hyperedge_attr, node_features


class HyperData(Data):
    """Offsets hyperedge_index rows separately when batching:
    row 0 (node ids) by num_nodes, row 1 (hyperedge ids) by num_hyperedges."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'hyperedge_index':
            return torch.tensor([[self.num_nodes], [self.hyperedge_attr.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class Dataset_ADNI_Hyper(Dataset):
    def __init__(self, data_dir, label_file, num_classes=2, k_neighbors=8, K_bins=5):
        self.data_dir = data_dir
        self.label_file = label_file
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        self.K_bins = K_bins
        self.data = []
        self.labels = []
        self._load()

    def _load(self):
        labels_df = pd.read_csv(self.label_file)
        labels_id = {'CN': 0, 'SMC': 0, 'EMCI': 0, 'LMCI': 1, 'AD': 1} if self.num_classes == 2 else \
                    {'CN': 0, 'SMC': 1, 'EMCI': 2, 'LMCI': 3, 'AD': 4}

        files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('sub-') and f.endswith('_aal.txt')])

        for fname in files:
            sid = fname.split('_')[0][4:]
            row = labels_df[labels_df['subject_id'] == sid]
            if row.empty:
                continue
            dx = row.iloc[0]['DX']
            if dx not in labels_id:
                continue

            try:
                ts = np.loadtxt(os.path.join(self.data_dir, fname))
            except Exception:
                continue

            if ts.ndim != 2 or ts.shape[1] != 116 or ts.shape[0] < 140:
                continue
            ts = ts[:140, :]

            hyperedge_index, hyperedge_attr, node_feat = build_hypergraph(
                ts, k_neighbors=self.k_neighbors, K_bins=self.K_bins
            )

            y = torch.tensor(labels_id[dx], dtype=torch.long)
            graph = HyperData(
                x=node_feat,
                hyperedge_index=hyperedge_index,
                hyperedge_attr=hyperedge_attr,
                y=y,
                num_nodes=node_feat.size(0)
            )
            self.data.append(graph)
            self.labels.append(labels_id[dx])

        print(f"Loaded {len(self.data)} hypergraphs | Class balance: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]