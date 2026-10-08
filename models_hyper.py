import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax, to_dense_batch


def signed_log(x):
    """Compresses heavy-tailed products of many z-scores; sign is preserved."""
    return torch.sign(x) * torch.log1p(x.abs())


def hyperedge_batch(hyperedge_index, batch, num_hyperedges):
    """Graph id of each hyperedge (all members share a graph)."""
    v, e = hyperedge_index
    eb = torch.zeros(num_hyperedges, dtype=torch.long, device=batch.device)
    eb[e] = batch[v]
    return eb


def get_batch(data):
    b = getattr(data, "batch", None)
    return b if b is not None else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)


# ═════════════════════════════════════════════════════════════════
# §3  Hypergraph message passing
# ═════════════════════════════════════════════════════════════════

class HyperCoFConv(nn.Module):
    """
    z_e       = σ( Σ_{v∈e} W_E x_v + W_f f_e )
    x_v^(l+1) = σ( Σ_{e∈E(v)} α_ve W_V z_e )      (+ residual + LayerNorm)
    α_ve over e∈E(v) from:  "attn" learned | "hcof" hCoF^B | "attn+hcof" both
    """
    def __init__(self, dim, f_dim, alpha_mode="attn+hcof"):
        super().__init__()
        assert alpha_mode in ("attn", "hcof", "attn+hcof")
        self.alpha_mode = alpha_mode
        self.W_E = nn.Linear(dim, dim, bias=False)    # no bias: a bias would be summed |e| times
        self.W_f = nn.Linear(f_dim, dim)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.att = nn.Linear(2 * dim, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))   # weight on hCoF^B in "attn+hcof"
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, hyperedge_index, f_e, hcof_top):
        v, e = hyperedge_index
        E, V = f_e.size(0), x.size(0)

        # node -> hyperedge  (SUM over members, as in the equation)
        z = F.gelu(scatter(self.W_E(x)[v], e, dim=0, dim_size=E, reduce="sum") + self.W_f(f_e))

        # α_ve, normalised over the hyperedges incident to v
        if self.alpha_mode == "hcof":
            logit = hcof_top[e]
        else:
            logit = F.leaky_relu(self.att(torch.cat([x[v], z[e]], -1)).squeeze(-1), 0.2)
            if self.alpha_mode == "attn+hcof":
                logit = logit + self.beta * hcof_top[e]
        alpha = softmax(logit, v, num_nodes=V)

        # hyperedge -> node  (SUM of α-weighted messages; α already sums to 1 per node)
        msg = scatter(alpha.unsqueeze(-1) * self.W_V(z)[e], v, dim=0, dim_size=V, reduce="sum")
        return self.norm(x + F.gelu(msg)), z


class BrainCHEF_HOC(nn.Module):
    def __init__(self, node_dim=116, f_dim=6, n_systems=1, hidden_dim=128, n_layers=2,
                 n_classes=2, dropout=0.3, alpha_mode="attn+hcof", log_scale=True):
        super().__init__()
        self.dropout, self.log_scale = dropout, log_scale
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList(HyperCoFConv(hidden_dim, f_dim, alpha_mode) for _ in range(n_layers))
        self.sca_proj = nn.Linear(n_systems * f_dim, hidden_dim)    # scaCoF readout
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))

    def forward(self, data):
        f_e, top, sca = data.hyperedge_attr, data.hcof_top, data.sca_attr.flatten(1)
        if self.log_scale:
            f_e, top, sca = signed_log(f_e), signed_log(top), signed_log(sca)

        x = self.input_proj(data.x)
        for conv in self.convs:
            x, z = conv(x, data.hyperedge_index, f_e, top)
            x = F.dropout(x, self.dropout, self.training)

        batch = get_batch(data)
        G = int(batch.max()) + 1
        eb = hyperedge_batch(data.hyperedge_index, batch, f_e.size(0))
        emb = torch.cat([scatter(x, batch, 0, dim_size=G, reduce="mean"),
                         scatter(z, eb, 0, dim_size=G, reduce="mean"),
                         self.sca_proj(sca)], -1)
        return self.classifier(emb), emb


# ═════════════════════════════════════════════════════════════════
# §4  Sequence Transformer alternative
# ═════════════════════════════════════════════════════════════════

class HyperTokenTransformer(nn.Module):
    """
    Tokens: [CLS] + node tokens + low-order hyperedge tokens (pairwise eCoF / triCoF)
            + one system scaffold token per system (scaCoF).
    Encodings: token type + system identity + hyperedge order + sequence position,
               with tokens ordered by parcellation system within each segment.
    """
    def __init__(self, node_dim, f_dim, n_systems, d=128, n_layers=4, n_heads=4, dropout=0.2,
                 n_classes=2, max_token_order=3, max_len=2048, log_scale=True):
        super().__init__()
        self.max_token_order, self.log_scale = max_token_order, log_scale
        self.node_in, self.he_in, self.sca_in = (nn.Linear(node_dim, d),
                                                 nn.Linear(f_dim, d), nn.Linear(f_dim, d))
        self.sys_emb = nn.Embedding(n_systems, d)
        self.type_emb = nn.Embedding(3, d)                 # 0 node | 1 hyperedge | 2 system
        self.order_emb = nn.Embedding(max_token_order + 1, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    @staticmethod
    def _order(graph_id, sys_id, S):
        # by graph, then system; stable sort keeps ROI/hyperedge index order inside a system
        return torch.sort(graph_id * S + sys_id, stable=True).indices

    def forward(self, data):
        batch = get_batch(data)
        G, S, dev = int(batch.max()) + 1, self.sys_emb.num_embeddings, batch.device
        f_e, sca = data.hyperedge_attr, data.sca_attr
        if self.log_scale:
            f_e, sca = signed_log(f_e), signed_log(sca)

        # node tokens
        p = self._order(batch, data.node_net, S)
        tv = (self.node_in(data.x[p]) + self.sys_emb(data.node_net[p]) + self.type_emb.weight[0])
        tv, mv = to_dense_batch(tv, batch[p], batch_size=G)

        # hyperedge tokens (orders 2..max_token_order)
        eb = hyperedge_batch(data.hyperedge_index, batch, f_e.size(0))
        k = (data.hyperedge_order <= self.max_token_order).nonzero().squeeze(-1)
        k = k[self._order(eb[k], data.hyperedge_net[k], S)]
        te = (self.he_in(f_e[k]) + self.sys_emb(data.hyperedge_net[k])
              + self.order_emb(data.hyperedge_order[k]) + self.type_emb.weight[1])
        te, me = to_dense_batch(te, eb[k], batch_size=G)

        # system scaffold tokens
        q = torch.arange(S, device=dev)
        ts = self.sca_in(sca) + self.sys_emb(q)[None] + self.type_emb.weight[2]
        ms = torch.ones(G, S, dtype=torch.bool, device=dev)

        seq = torch.cat([self.cls.expand(G, -1, -1), tv, te, ts], 1)
        mask = torch.cat([torch.ones(G, 1, dtype=torch.bool, device=dev), mv, me, ms], 1)
        L = seq.size(1)
        assert L <= self.pos_emb.num_embeddings, f"sequence length {L} > max_len"
        seq = seq + self.pos_emb(torch.arange(L, device=dev))[None]

        h = self.encoder(seq, src_key_padding_mask=~mask)[:, 0]
        return self.head(h), h