import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax

class HypergraphCoFluctuationConv(nn.Module):
    """
    Cross-level message passing:
      1. Node -> Hyperedge: Nodes pass embeddings into hyperedge, 
                            fused with higher-order co-fluctuation attributes (hDYN + hCoF).
      2. Hyperedge -> Node: Hyperedges send refined representations back to member nodes
                            via learned attention.
    """
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        self.node_proj = nn.Linear(in_channels, out_channels)
        self.edge_proj = nn.Linear(edge_dim, out_channels)
        
        # Attention from hyperedge to member nodes
        self.attn_linear = nn.Linear(out_channels * 2, 1)
        self.update_node = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, hyperedge_index, hyperedge_attr):
        """
        x: (total_nodes, in_channels)
        hyperedge_index: (2, num_incidences) [node_idx, hyperedge_idx]
        hyperedge_attr:  (num_hyperedges, edge_dim)
        """
        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]

        # ── Step 1: Project features ──
        h_node = self.node_proj(x)                  # (V, out_channels)
        h_edge = self.edge_proj(hyperedge_attr)      # (E, out_channels)

        # ── Step 2: Node -> Hyperedge Aggregation ──
        inc_node_feat = h_node[node_idx]             # Gather features of member nodes
        # Aggregate member nodes into hyperedge
        edge_msg = scatter_mean(inc_node_feat, edge_idx, dim=0, dim_size=hyperedge_attr.size(0))
        # Fuse with hyperedge co-fluctuation profile
        edge_repr = F.gelu(edge_msg + h_edge)        # (E, out_channels)

        # ── Step 3: Hyperedge -> Node Attention Update ──
        inc_edge_feat = edge_repr[edge_idx]          # (num_incidences, out_channels)
        pair_feat = torch.cat([inc_node_feat, inc_edge_feat], dim=-1)
        alpha = F.leaky_relu(self.attn_linear(pair_feat), negative_slope=0.2)
        alpha = scatter_softmax(alpha, node_idx, dim=0)

        # Weighted message back to nodes
        node_msg = scatter_mean(alpha * inc_edge_feat, node_idx, dim=0, dim_size=x.size(0))
        out = self.norm(x + self.update_node(node_msg))
        return out, edge_repr


class BrainCHEF_HOC(nn.Module):
    """
    Hypergraph model fusing Node Functional Profiles with 
    Higher-Order Hyperedge Co-fluctuations (Triangles, k-NNs, and Yeo Systems).
    """
    def __init__(self, node_dim=116, edge_dim=6, hidden_dim=128, n_layers=2, n_classes=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Feature transformations
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList([
            HypergraphCoFluctuationConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim, 
                edge_dim=edge_dim
            ) for _ in range(n_layers)
        ])

        # Dual readout: combines node-level pooling and hyperedge-level pooling
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        hyperedge_index = data.hyperedge_index
        hyperedge_attr = data.hyperedge_attr

        # Pass through multi-layer hypergraph co-fluctuation convs
        edge_repr = None
        for conv in self.convs:
            x, edge_repr = conv(x, hyperedge_index, hyperedge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Global Readout ──
        # Handle batching across multiple subjects
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            num_graphs = batch.max().item() + 1
            node_pool = scatter_mean(x, batch, dim=0, dim_size=num_graphs)
        else:
            node_pool = x.mean(dim=0, keepdim=True)

        # Graph-level pooling of hyperedges (via node assignment mapping)
        edge_batch = data.batch[hyperedge_index[0]] if hasattr(data, 'batch') and data.batch is not None else None
        if edge_batch is not None:
            # Map edge incidences to graph indices
            e_graph_idx = scatter_mean(edge_batch.float(), hyperedge_index[1], dim=0).long()
            edge_pool = scatter_mean(edge_repr, e_graph_idx, dim=0, dim_size=num_graphs)
        else:
            edge_pool = edge_repr.mean(dim=0, keepdim=True)

        combined = torch.cat([node_pool, edge_pool], dim=-1)
        logits = self.classifier(combined)
        return logits, combined