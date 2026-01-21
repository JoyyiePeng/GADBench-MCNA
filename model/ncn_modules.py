"""
Multi-hop Common Neighbor (CN) Computation Module
"""

import os
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
from scipy.sparse import csr_matrix
import math


# ============ Global Cache Management ============
_MULTIHOP_CN_CACHE = {}  # Multi-hop cache


def _get_multihop_hash(graph, max_hops, cn_threshold):
    """Compute unique identifier for multi-hop graph"""
    src, dst = graph.edges()
    edge_str = f"{graph.num_nodes()}_{graph.num_edges()}_{src.sum().item():.0f}_{dst.sum().item():.0f}_hops{max_hops}_th{cn_threshold}"
    return hashlib.md5(edge_str.encode()).hexdigest()[:12]


# ============================================================
# Multi-hop CN Computer
# ============================================================

class MultiHopCNComputer:
    """
    Multi-hop Common Neighbor Pre-computer

    Computes CN attention matrices for 1-hop, 2-hop, 3-hop:
    - 1-hop: A^1 (direct neighbors)
    - 2-hop: A^2 (traditional Common Neighbors)
    - 3-hop: A^3 (3-hop reachable)
    """

    def __init__(self,
                 max_hops: int = 3,
                 cn_threshold: int = 1,
                 max_nodes_dense: int = 10000,
                 cache_dir: str = ".cn_cache",
                 use_file_cache: bool = True):
        self.max_hops = max_hops
        self.cn_threshold = cn_threshold
        self.max_nodes_dense = max_nodes_dense
        self.cache_dir = cache_dir
        self.use_file_cache = use_file_cache

        # Cache data
        self._cn_data_list = None  # list of hop data
        self._is_precomputed = False
        self._use_dense = False
        self._cache_device = None

    def precompute(self, graph, device):
        """Precompute CN matrices for all hops"""
        if self._is_precomputed:
            self._ensure_device(device)
            return

        graph_hash = _get_multihop_hash(graph, self.max_hops, self.cn_threshold)

        # 1. Check memory cache
        if graph_hash in _MULTIHOP_CN_CACHE:
            print(f"[MultiHop CN] Loading from memory: {graph_hash}")
            self._load_from_cache(_MULTIHOP_CN_CACHE[graph_hash], device)
            return

        # 2. Check file cache
        if self.use_file_cache:
            cache_path = os.path.join(self.cache_dir, f"multihop_cn_{graph_hash}.pt")
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(cache_path):
                print(f"[MultiHop CN] Loading from file: {cache_path}")
                cn_data = torch.load(cache_path, map_location='cpu')
                _MULTIHOP_CN_CACHE[graph_hash] = cn_data
                self._load_from_cache(cn_data, device)
                return

        # 3. Compute
        print(f"[MultiHop CN] Computing... (nodes={graph.num_nodes()}, hops={self.max_hops})")
        cn_data = self._compute_all_hops(graph, device)

        # 4. Save
        _MULTIHOP_CN_CACHE[graph_hash] = cn_data
        if self.use_file_cache:
            cache_path = os.path.join(self.cache_dir, f"multihop_cn_{graph_hash}.pt")
            torch.save(self._to_cpu(cn_data), cache_path)
            print(f"[MultiHop CN] Saved: {cache_path}")

        self._load_from_cache(cn_data, device)

    def _compute_all_hops(self, graph, device):
        """Compute CN for all hops"""
        num_nodes = graph.num_nodes()
        use_dense = num_nodes <= self.max_nodes_dense

        if graph.num_edges() == 0:
            return {'empty': True, 'use_dense': use_dense, 'hop_data': []}

        if use_dense:
            return self._compute_dense(graph, device, num_nodes)
        else:
            return self._compute_sparse(graph, device, num_nodes)

    def _compute_dense(self, graph, device, num_nodes):
        """Dense mode computation"""
        adj = graph.adjacency_matrix().to_dense().to(device).float()

        hop_data = []
        adj_power = torch.eye(num_nodes, device=device)


        for hop in range(1, self.max_hops + 1):
            adj_power = torch.mm(adj_power, adj)

            cn_matrix = adj_power.clone()
            cn_matrix.fill_diagonal_(0)


            # ========== Top-K Sparsification ==========
            if hop >= 2:  # Only for 2-hop and above
                k = min(50, num_nodes - 1)  # Keep at most 50 neighbors per node

                # Find top-k per row
                top_values, top_indices = torch.topk(cn_matrix, k=k, dim=1)

                # Build sparsified matrix
                cn_matrix_sparse = torch.zeros_like(cn_matrix)
                cn_matrix_sparse.scatter_(1, top_indices, top_values)
                cn_matrix = cn_matrix_sparse
            # ========================================

            cn_matrix = cn_matrix * (cn_matrix >= self.cn_threshold).float()

            has_cn = (cn_matrix.sum(dim=1, keepdim=True) > 0).float()

            if has_cn.sum() > 0:
                attn_scores = torch.log1p(cn_matrix)
                attn_scores = attn_scores.masked_fill(cn_matrix == 0, -1e9)
                attn_matrix = F.softmax(attn_scores, dim=1)
                attn_matrix[has_cn.squeeze() == 0] = 0
            else:
                attn_matrix = torch.zeros_like(cn_matrix)

            hop_data.append({
                'attn_matrix': attn_matrix,
                'has_cn': has_cn,
            })
            print(f"  [Hop-{hop}] Valid nodes: {int(has_cn.sum().item())}/{num_nodes}")

        return {'empty': False, 'use_dense': True, 'hop_data': hop_data}

    def _compute_sparse(self, graph, device, num_nodes):
        """Sparse mode computation"""
        src, dst = graph.edges()
        src_np, dst_np = src.cpu().numpy(), dst.cpu().numpy()

        data = np.ones(len(src_np), dtype=np.float32)
        adj = csr_matrix((data, (src_np, dst_np)), shape=(num_nodes, num_nodes))

        hop_data = []
        adj_power = csr_matrix(np.eye(num_nodes, dtype=np.float32))



        for hop in range(1, self.max_hops + 1):
            adj_power = adj_power @ adj

            cn_sparse = adj_power.copy()
            cn_sparse.setdiag(0)
            cn_sparse.eliminate_zeros()

            # ========== Top-K Sparsification ==========
            if hop >= 2:
                k = 50
                cn_csr = cn_sparse.tocsr()

                # Keep top-k per row
                for i in range(num_nodes):
                    row_start, row_end = cn_csr.indptr[i], cn_csr.indptr[i+1]
                    if row_end - row_start > k:
                        row_data = cn_csr.data[row_start:row_end]
                        row_indices = cn_csr.indices[row_start:row_end]

                        # Find top-k
                        top_k_idx = np.argpartition(row_data, -k)[-k:]

                        # Zero out non-top-k
                        mask = np.ones(len(row_data), dtype=bool)
                        mask[top_k_idx] = False
                        cn_csr.data[row_start:row_end][mask] = 0

                cn_sparse = cn_csr
                cn_sparse.eliminate_zeros()
            # ========================================
            cn_coo = cn_sparse.tocoo()
            mask = cn_coo.data >= self.cn_threshold
            rows, cols = cn_coo.row[mask], cn_coo.col[mask]
            cn_values = cn_coo.data[mask]

            if len(cn_values) == 0:
                hop_data.append({
                    'cn_graph': None,
                    'attn_weights': None,
                    'has_cn': torch.zeros(num_nodes, 1, device=device),
                })
                print(f"  [Hop-{hop}] No valid CN")
                continue

            cn_graph = dgl.graph((rows, cols), num_nodes=num_nodes, device=device)
            cn_weights = torch.tensor(cn_values, dtype=torch.float32, device=device)

            scores = torch.log1p(cn_weights)
            attn_weights = dgl.ops.edge_softmax(cn_graph, scores)
            has_cn = (cn_graph.in_degrees() > 0).float().unsqueeze(1)

            hop_data.append({
                'cn_graph': cn_graph,
                'attn_weights': attn_weights,
                'has_cn': has_cn,
            })
            print(f"  [Hop-{hop}] CN edges: {cn_graph.num_edges()}")

        return {'empty': False, 'use_dense': False, 'hop_data': hop_data}

    def _load_from_cache(self, cn_data, device):
        """Load from cache"""
        if cn_data.get('empty', True):
            self._cn_data_list = []
            self._is_precomputed = True
            return

        self._use_dense = cn_data['use_dense']
        self._cn_data_list = []

        for hop_info in cn_data['hop_data']:
            if self._use_dense:
                self._cn_data_list.append({
                    'attn_matrix': hop_info['attn_matrix'].to(device),
                    'has_cn': hop_info['has_cn'].to(device),
                })
            else:
                if hop_info['cn_graph'] is not None:
                    self._cn_data_list.append({
                        'cn_graph': hop_info['cn_graph'].to(device),
                        'attn_weights': hop_info['attn_weights'].to(device),
                        'has_cn': hop_info['has_cn'].to(device),
                    })
                else:
                    self._cn_data_list.append({
                        'cn_graph': None,
                        'attn_weights': None,
                        'has_cn': hop_info['has_cn'].to(device),
                    })

        self._cache_device = device
        self._is_precomputed = True

    def _to_cpu(self, cn_data):
        """Move to CPU"""
        if cn_data.get('empty', True):
            return cn_data

        hop_data_cpu = []
        for hop_info in cn_data['hop_data']:
            if cn_data['use_dense']:
                hop_data_cpu.append({
                    'attn_matrix': hop_info['attn_matrix'].cpu(),
                    'has_cn': hop_info['has_cn'].cpu(),
                })
            else:
                if hop_info['cn_graph'] is not None:
                    hop_data_cpu.append({
                        'cn_graph': hop_info['cn_graph'].to('cpu'),
                        'attn_weights': hop_info['attn_weights'].cpu(),
                        'has_cn': hop_info['has_cn'].cpu(),
                    })
                else:
                    hop_data_cpu.append({
                        'cn_graph': None,
                        'attn_weights': None,
                        'has_cn': hop_info['has_cn'].cpu(),
                    })

        return {'empty': False, 'use_dense': cn_data['use_dense'], 'hop_data': hop_data_cpu}

    def _ensure_device(self, device):
        """Ensure correct device"""
        if self._cache_device == device:
            return

        for i, hop_info in enumerate(self._cn_data_list):
            if self._use_dense:
                self._cn_data_list[i] = {
                    'attn_matrix': hop_info['attn_matrix'].to(device),
                    'has_cn': hop_info['has_cn'].to(device),
                }
            else:
                if hop_info['cn_graph'] is not None:
                    self._cn_data_list[i] = {
                        'cn_graph': hop_info['cn_graph'].to(device),
                        'attn_weights': hop_info['attn_weights'].to(device),
                        'has_cn': hop_info['has_cn'].to(device),
                    }
        self._cache_device = device

    def aggregate(self, feat, hop_idx, transform_fn=None):
        """
        Aggregate features using CN matrix of specified hop

        Args:
            feat: [N, d] node features
            hop_idx: hop index (0=1-hop, 1=2-hop, 2=3-hop)
            transform_fn: feature transformation function
        """
        h = transform_fn(feat) if transform_fn else feat

        if hop_idx >= len(self._cn_data_list):
            return h

        hop_data = self._cn_data_list[hop_idx]

        if self._use_dense:
            attn_matrix = hop_data['attn_matrix']
            has_cn = hop_data['has_cn']

            out = torch.mm(attn_matrix, h)
            out = has_cn * out + (1 - has_cn) * h
            return out
        else:
            cn_graph = hop_data['cn_graph']
            has_cn = hop_data['has_cn']

            if cn_graph is None:
                return h

            with cn_graph.local_scope():
                cn_graph.ndata['h'] = h
                cn_graph.edata['attn'] = hop_data['attn_weights']
                cn_graph.update_all(
                    fn.u_mul_e('h', 'attn', 'm'),
                    fn.sum('m', 'out')
                )
                out = cn_graph.ndata['out']
                out = has_cn * out + (1 - has_cn) * h
            return out

    @property
    def num_hops(self):
        return len(self._cn_data_list) if self._cn_data_list else 0


# ============================================================
# Gated Fusion Module
# ============================================================

class GatedFusion(nn.Module):
    """Node-level gated fusion with range constraints"""
    def __init__(self, hidden_dim, dropout=0.1,
                 min_gcn_weight=0.1, max_gcn_weight=1, gcn_init_weight=0.5):
        super().__init__()
        # GCN weight range [min, max], MoE automatically [1-max, 1-min]
        self.min_gcn_weight = min_gcn_weight
        self.max_gcn_weight = max_gcn_weight
        self.temperature = nn.Parameter(torch.ones(1))
        self.gcn_init_weight = gcn_init_weight
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Output only 1 value
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate[-1].weight, std=0.01)
        # Make sigmoid output close to gcn_init_weight position in [min, max] range
        init_ratio = (self.gcn_init_weight - self.min_gcn_weight) / (self.max_gcn_weight - self.min_gcn_weight + 1e-8)
        init_ratio = max(0.01, min(0.99, init_ratio))  # Avoid extreme values
        init_bias = math.log(init_ratio / (1 - init_ratio + 1e-8))
        self.gate[-1].bias.data = torch.tensor([init_bias])

    def forward(self, gcn_out, moe_out):
        gate_input = torch.cat([gcn_out, moe_out], dim=-1)
        logit = self.gate(gate_input)  # [N, 1]

        # sigmoid output in [0, 1], map to [min_gcn, max_gcn]
        ratio = torch.sigmoid(logit)
        gcn_weight = self.min_gcn_weight + (self.max_gcn_weight - self.min_gcn_weight) * ratio
        moe_weight = 1 - gcn_weight

        out = gcn_weight * gcn_out + moe_weight * moe_out

        weights = torch.cat([gcn_weight, moe_weight], dim=-1)
        return out, weights


# ============================================================
# Sparse Multi-hop MoE Layer
# ============================================================

class SparseMultiHopMoE(nn.Module):
    """
    Sparse Multi-hop MoE Layer

    Architecture:
    - Main path: GCN (always selected, not routed)
    - Enhancement path: 1-hop, 2-hop, 3-hop CN (sparse top-k selection)
    """

    def __init__(self,
                 gcn_layer,
                 in_feats: int,
                 out_feats: int,
                 num_hops: int = 3,
                 top_k: int = 1,
                 dropout: float = 0.5,
                 noise_std: float = 0.2,
                 load_balance_weight: float = 0.01,
                 cn_threshold: int = 1,
                 max_nodes_dense: int = 10000,
                 router_temperature=1.0,
                 fixed_hop=None,
                 cache_dir: str = ".cn_cache"):
        super().__init__()

        self.gcn_layer = gcn_layer
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_hops = num_hops
        self.top_k = min(top_k, num_hops)
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight
        self.router_temperature = nn.Parameter(torch.tensor(router_temperature))
        self.min_temperature = 2.0  # Minimum temperature
        self.fixed_hop = fixed_hop

        # Multi-hop CN computer
        self.cn_computer = MultiHopCNComputer(
            max_hops=num_hops,
            cn_threshold=cn_threshold,
            max_nodes_dense=max_nodes_dense,
            cache_dir=cache_dir,
        )

        # Transform layer for each hop
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_feats, out_feats) for _ in range(num_hops)
        ])

        # Router: looks at GCN output + each hop output
        self.router = nn.Sequential(
            nn.Linear(out_feats * (num_hops + 1), out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, num_hops)
        )

        # Overall enhancement coefficient
        self.alpha = nn.Parameter(torch.zeros(1))

        # Statistics
        self.aux_loss = 0.0
        self._last_expert_stats = {}

        self.input_proj = nn.Linear(in_feats, out_feats) if in_feats != out_feats else nn.Identity()

        # Gated fusion layer
        self.fusion = GatedFusion(out_feats, dropout=0.3)
        self.layer_norm = nn.LayerNorm(out_feats)

    def forward(self, graph, feat, main_out=None):
        device = feat.device

        h_original = feat
        # Precompute multi-hop CN
        self.cn_computer.precompute(graph, device)

        # ===== Main path =====
        if main_out is None:
            main_out = self.gcn_layer(graph, feat)  # Original behavior


        # ===== Compute gains for all hops =====
        hop_gains = []
        for i in range(self.num_hops):
            gain = self.cn_computer.aggregate(
                feat, hop_idx=i, transform_fn=self.hop_transforms[i]
            )
            hop_gains.append(gain)


        # ===== Key: concatenate router input first, then stack =====
        router_input = torch.cat([main_out] + hop_gains, dim=-1)  # [N, out_feats * (num_hops+1)]
        router_logits = self.router(router_input)  # [N, num_hops]

        # ===== Temperature scaling (consistent with minibatch) =====
        temp = F.softplus(self.router_temperature) + self.min_temperature
        router_logits = router_logits / temp
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # Now stack for subsequent weighting
        hop_gains = torch.stack(hop_gains, dim=1)  # [N, num_hops, out_feats]


        # Top-k selection
        if self.top_k < self.num_hops:
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)

            sparse_weights = torch.zeros_like(router_logits)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            router_probs = sparse_weights
        else:
            router_probs = F.softmax(router_logits, dim=-1)


        if self.training:
            router_probs = F.dropout(router_probs, p=0.1, training=True)

        if self.fixed_hop is not None:
            moe_out = hop_gains[:, self.fixed_hop, :]
        else:
            moe_out = (hop_gains * router_probs.unsqueeze(-1)).sum(dim=1)  # [N, out_feats]

        # ===== Gated fusion =====
        output, gate_weights = self.fusion(main_out, moe_out)

        # Residual + LayerNorm
        output = self.layer_norm(output)


        # ===== Auxiliary loss =====
        if self.training:
            self._compute_aux_loss(router_logits, router_probs)

        # Statistics
        with torch.no_grad():
            # Record gate weights
            self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
            self._last_expert_stats = {
                'gcn_weight': gate_weights[:, 0].mean().item(),
                'moe_weight': gate_weights[:, 1].mean().item(),
            }
            for i in range(self.num_hops):
                self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()

        return output

    def _compute_aux_loss(self, router_logits, router_probs):
        """Load balancing loss"""
        expert_usage = router_probs.mean(dim=0)
        router_prob_avg = F.softmax(router_logits, dim=-1).mean(dim=0)
        self.aux_loss = self.load_balance_weight * self.num_hops * (expert_usage * router_prob_avg).sum()


    def get_aux_loss(self):
        return self.aux_loss

    def get_expert_stats(self):
        return self._last_expert_stats

    def precompute_all(self, graph, feat):
        """
        Precompute CN aggregation results for all nodes on full graph (without applying transform)

        Args:
            graph: Full graph
            feat: [N, in_feats] Original features of full graph
        """
        device = feat.device

        # Reuse cn_computer.precompute
        self.cn_computer.precompute(graph, device)

        # Precompute aggregation results for each hop (without transform to avoid computation graph issues)
        self._precomputed_hop_feats = []
        with torch.no_grad():  # Key: don't create computation graph
            for i in range(self.num_hops):
                # Only do CN aggregation, don't apply hop_transforms
                hop_feat = self.cn_computer.aggregate(feat, hop_idx=i, transform_fn=None)
                self._precomputed_hop_feats.append(hop_feat)

        print(f"[MoE] Precomputation complete: {self.num_hops} hops")



    def forward_minibatch(self, feat, node_indices, main_out=None, transformed_feat=None):
        """
        Mini-batch forward: index precomputed CN features

        Args:
            feat: [batch_size, in_feats] Original features of this batch
            node_indices: [batch_size] Node indices in full graph
            main_out: [batch_size, out_feats] Externally provided main path output (e.g., DGA convolution)

        Returns:
            [batch_size, out_feats]
        """
        h_original = feat

        # ===== Main path =====
        if main_out is None:
            # When no external input, use projection (maintain original behavior)
            main_out = self.input_proj(feat)

        # ===== Index precomputed hop features and apply transform =====
        hop_feats_raw = [
            self._precomputed_hop_feats[i][node_indices]
            for i in range(self.num_hops)
        ]
        hop_gains = [
            self.hop_transforms[i](hop_feats_raw[i])
            for i in range(self.num_hops)
        ]

        # ===== Routing (consistent with forward) =====
        router_input = torch.cat([main_out] + hop_gains, dim=-1)
        router_logits = self.router(router_input)
        # ===== Temperature scaling (higher temperature → more uniform) =====
        temp = F.softplus(self.router_temperature) + self.min_temperature
        router_logits = router_logits / temp
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        # Top-k selection
        if self.top_k < self.num_hops:
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)
            sparse_weights = torch.zeros_like(router_logits)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            router_probs = sparse_weights
        else:
            router_probs = F.softmax(router_logits, dim=-1)

        if self.training:
            router_probs = F.dropout(router_probs, p=0.1, training=True)

        # ===== Weighted aggregation (consistent with forward) =====
        hop_gains_stack = torch.stack(hop_gains, dim=1)
        if self.fixed_hop is not None:
            moe_out = hop_gains_stack[:, self.fixed_hop, :]
        else:
            moe_out = hop_gains_stack.mean(dim=1)

        # ===== Key modification: use externally passed transformed_feat =====
        if transformed_feat is not None:
            h_proj = transformed_feat
        else:
            h_proj=h_original
        output, gate_weights = self.fusion(main_out, moe_out)  # Two-way fusion
        output = self.layer_norm(output)

        # ===== Auxiliary loss =====
        if self.training:
            self._compute_aux_loss(router_logits, router_probs)

        # ===== Statistics (consistent with forward) =====
        with torch.no_grad():
            self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
            self._last_expert_stats = {
                'gcn_weight': gate_weights[:, 0].mean().item(),
                'moe_weight': gate_weights[:, 1].mean().item(),
            }
            for i in range(self.num_hops):
                self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()

        return output
