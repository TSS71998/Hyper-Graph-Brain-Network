import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from itertools import combinations
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────
# HOC Co-Fluctuation & RSS Binning Utilities
# ─────────────────────────────────────────────────────────────────

def zscore(ts, eps=1e-8):
    """ts: (T, N) -> column-wise z-score."""
    return (ts - ts.mean(0, keepdims=True)) / (ts.std(0, keepdims=True) + eps)

def hyperedge_ts(ts_z, hyperedges):
    """hTS_e(t) = prod_{v in e} z_v(t).   ts_z: (T, N) -> (E, T)"""
    return np.stack([ts_z[:, sorted(e)].prod(axis=1) for e in hyperedges]).astype(np.float32)

def h_dyn(sig, w=20):
    """hDYN_e = Var_tau( (1/w) * sum_{t=tau}^{tau+w-1} hTS_e(t) ),
    over every tau = 0..T-w (stride 1, as in the equation).  sig: (M, T) -> (M, 1)"""
    M, T = sig.shape
    if T < w:
        return np.zeros((M, 1), np.float32)
    c = np.cumsum(np.pad(sig, ((0, 0), (1, 0))), axis=1)   # c[:, k] = sum of first k samples
    win_mean = (c[:, w:] - c[:, :-w]) / w                    # (M, T-w+1)
    return win_mean.var(axis=1, keepdims=True).astype(np.float32)

def rss(hts):
    """RSS(t) = sqrt( sum_{e in E} hTS_e(t)^2 ).  (E, T) -> (T,)"""
    return np.sqrt((hts ** 2).sum(axis=0))

def rss_bins(hts, B=5):
    """Sort timepoints by RSS and split into B bins T_1..T_B (low -> high amplitude).
    array_split keeps every timepoint even when T is not divisible by B."""
    return np.array_split(np.argsort(rss(hts)), B)

def h_cof(sig, bins):
    """hCoF_e^b = (1/|T_b|) * sum_{t in T_b} hTS_e(t).  (M, T) -> (M, B)"""
    return np.stack([sig[:, tb].mean(axis=1) for tb in bins], axis=1).astype(np.float32)



# ═════════════════════════════════════════════════════════════════
# Candidate hypergraph: order 2 (pairs), order 3 (triangles), k-NN groups
# ═════════════════════════════════════════════════════════════════

def candidate_hyperedges(fc, k_neighbors=8, n_tri_cands=5, max_tri_per_roi=3, edge_density=0.10):
    N = fc.shape[0]
    a = np.abs(fc.copy())
    np.fill_diagonal(a, -1.0)
    seen, hyperedges = set(), []

    def add(e):
        e = frozenset(int(v) for v in e)
        if len(e) >= 2 and e not in seen:
            seen.add(e)
            hyperedges.append(e)

    # order 2: strongest |FC| pairs (the pairwise level the scaffold is defined on)
    iu = np.triu_indices(N, 1)
    n_keep = max(1, int(edge_density * len(iu[0])))
    for k in np.argsort(a[iu])[-n_keep:]:
        add((iu[0][k], iu[1][k]))

    # order 3: strongest triangles anchored at each ROI
    for i in range(N):
        cands = np.argsort(a[i])[-n_tri_cands:]
        tris = sorted(((a[i, u] + a[i, v] + a[u, v], (i, u, v))
                       for u, v in combinations(cands, 2)), reverse=True)
        for _, tri in tris[:max_tri_per_roi]:
            add(tri)

    # order k+1: k-NN neighbourhoods
    for i in range(N):
        add([i, *np.argsort(a[i])[-k_neighbors:]])

    return hyperedges

def scaffold_weights(hyperedges, mode="next_order"):
    """w_e^sca = number of larger hyperedges that contain e.
    mode="next_order": count only hyperedges of the next size up (pairs in triangles,
                       triangles in k-NN groups).
    mode="any_larger": count every strictly larger superset."""
    sizes = sorted({len(e) for e in hyperedges})
    nxt = {s: sizes[i + 1] for i, s in enumerate(sizes[:-1])}
    w = np.zeros(len(hyperedges), np.float32)
    for i, e in enumerate(hyperedges):
        if len(e) not in nxt:
            continue                       # top order: nothing larger to sit inside
        for f in hyperedges:
            size_ok = len(f) == nxt[len(e)] if mode == "next_order" else len(f) > len(e)
            if size_ok and e < f:          # frozenset '<' is proper subset
                w[i] += 1
    return w


def scaffold_filter(hyperedges, w, tau=1):
    """Keep lower-order hyperedges with w_e^sca >= tau. Top-order hyperedges have no
    larger hyperedges to be counted in, so they are always kept."""
    max_size = max(len(e) for e in hyperedges)
    keep = [i for i, e in enumerate(hyperedges) if len(e) == max_size or w[i] >= tau]
    return [hyperedges[i] for i in keep], w[keep]


def hyperedge_system(hyperedges, roi2net, Q):
    """Assign each hyperedge to the system holding the majority of its members."""
    return np.array([np.bincount(roi2net[sorted(e)], minlength=Q).argmax()
                     for e in hyperedges], dtype=np.int64)


def scaffold_system_ts(hts, w, e_net, Q):
    """scaTS_q(t) = (1/|E_q^sca|) * sum_{e in E_q^sca} w_e^sca * hTS_e(t).  -> (Q, T)
    E_q^sca = scaffold hyperedges (w_e > 0) assigned to system q."""
    out = np.zeros((Q, hts.shape[1]), np.float32)
    members = defaultdict(list)
    for i in np.flatnonzero(w > 0):
        members[int(e_net[i])].append(i)
    for q, idx in members.items():
        out[q] = (w[idx, None] * hts[idx]).sum(0) / len(idx)
    return out

# ─────────────────────────────────────────────────────────────────
# Hypergraph Builder
# ─────────────────────────────────────────────────────────────────

def build_hypergraph(ts, roi2net=None, k_neighbors=8, K_bins=5, window=20,
                     edge_density=0.10, sca_tau=1, sca_mode="next_order"):
    T, N = ts.shape
    ts_z = zscore(ts)
    fc = np.nan_to_num(np.corrcoef(ts_z.T))
    roi2net = np.zeros(N, np.int64) if roi2net is None else np.asarray(roi2net, np.int64)
    Q = int(roi2net.max()) + 1

    # §2 scaffold: weight candidates, then filter
    cand = candidate_hyperedges(fc, k_neighbors, edge_density=edge_density)
    w_all = scaffold_weights(cand, sca_mode)
    hyperedges, w_sca = scaffold_filter(cand, w_all, sca_tau)

    # §1 dynamics on the retained hypergraph E
    hts = hyperedge_ts(ts_z, hyperedges)                       # (E, T)
    bins = rss_bins(hts, K_bins)                               # T_1..T_B
    f_e = np.concatenate([h_dyn(hts, window),
                          h_cof(hts, bins)], axis=1)           # f_e = [hDYN, hCoF^1..B]

    # §2 system-level scaffold profiles (same RSS bins as the hyperedges)
    e_net = hyperedge_system(hyperedges, roi2net, Q)
    sca_ts = scaffold_system_ts(hts, w_sca, e_net, Q)          # (Q, T)
    sca_f = np.concatenate([h_dyn(sca_ts, window),
                            h_cof(sca_ts, bins)], axis=1)      # (Q, 1+B)

    # incidence H as (node, hyperedge) pairs
    inc = [(v, i) for i, e in enumerate(hyperedges) for v in sorted(e)]

    return dict(
        x=torch.tensor(fc, dtype=torch.float32),
        hyperedge_index=torch.tensor(inc, dtype=torch.long).t().contiguous(),
        hyperedge_attr=torch.tensor(f_e),                          # (E, 1+B)
        hcof_top=torch.tensor(f_e[:, -1]),                         # hCoF^B, highest-RSS bin
        scaffold_w=torch.tensor(w_sca),                            # (E,)
        hyperedge_order=torch.tensor([len(e) for e in hyperedges]),
        hyperedge_net=torch.tensor(e_net),
        node_net=torch.tensor(roi2net),
        sca_attr=torch.tensor(sca_f).unsqueeze(0),                 # (1, Q, 1+B), batches to (G, Q, 1+B)
    )


class HyperData(Data):
    """Offsets hyperedge_index rows separately when batching:
    row 0 (node ids) by num_nodes, row 1 (hyperedge ids) by num_hyperedges."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'hyperedge_index':
            return torch.tensor([[self.num_nodes], [self.hyperedge_attr.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


class Dataset_ADNI_Hyper(Dataset):
    def __init__(self, data_dir, label_file, num_classes=2, k_neighbors=8, K_bins=5,
                 roi2net_file=None, window=20, edge_density=0.10, sca_tau=1):
        self.data_dir = data_dir
        self.label_file = label_file
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        self.K_bins = K_bins
        self.data = []
        self.labels = []
        self.roi2net = np.loadtxt(roi2net_file, dtype=np.int64) if roi2net_file else None
        self.window, self.edge_density, self.sca_tau = window, edge_density, sca_tau
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

            g = build_hypergraph(ts, roi2net=self.roi2net, k_neighbors=self.k_neighbors,
                                 K_bins=self.K_bins, window=self.window,
                                 edge_density=self.edge_density, sca_tau=self.sca_tau)
            graph = HyperData(**g, y=torch.tensor(labels_id[dx], dtype=torch.long),
                              num_nodes=g['x'].size(0))

            y = torch.tensor(labels_id[dx], dtype=torch.long)
            self.data.append(graph)
            self.labels.append(labels_id[dx])

        print(f"Loaded {len(self.data)} hypergraphs | Class balance: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]