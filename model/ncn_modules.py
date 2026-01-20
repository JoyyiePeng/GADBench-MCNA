"""
NCN Layer with Common Neighbor Attention
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


class LearnableZScoreNorm(nn.Module):
    """可学习的 Z-Score 归一化，类似 BatchNorm 但按特征维度"""
    def __init__(self, num_features, eps=1e-8, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        x_norm = (x - mean) / (std + self.eps)
        
        if self.affine:
            x_norm = self.gamma * x_norm + self.beta
        
        return x_norm


# ============ 全局缓存管理 ============
_CN_CACHE = {}  # 内存缓存: {graph_hash: cn_data}
_MULTIHOP_CN_CACHE = {}  # 多跳缓存


def _get_graph_hash(graph, cn_threshold):
    """计算图的唯一标识"""
    src, dst = graph.edges()
    edge_str = f"{graph.num_nodes()}_{graph.num_edges()}_{src.sum().item():.0f}_{dst.sum().item():.0f}_th{cn_threshold}"
    return hashlib.md5(edge_str.encode()).hexdigest()[:12]


def _get_multihop_hash(graph, max_hops, cn_threshold):
    """计算多跳图的唯一标识"""
    src, dst = graph.edges()
    edge_str = f"{graph.num_nodes()}_{graph.num_edges()}_{src.sum().item():.0f}_{dst.sum().item():.0f}_hops{max_hops}_th{cn_threshold}"
    return hashlib.md5(edge_str.encode()).hexdigest()[:12]


def _get_cache_path(graph_hash, cache_dir=".cn_cache"):
    """获取缓存文件路径"""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"cn_{graph_hash}.pt")


class NCNLayer(nn.Module):
    """
    NCN Layer - 支持自动文件缓存
    
    特点：
    - 第一次 forward 自动预计算并缓存到文件
    - 之后的运行自动从文件加载
    - 同一数据跑多次完全复用
    """
    
    def __init__(self, 
                 in_feats: int,
                 out_feats: int,
                 dropout: float = 0.1,
                 activation: str = 'ReLU',
                 cn_threshold: int = 1,
                 max_nodes_dense: int = 10000,
                 cache_dir: str = ".cn_cache",
                 use_file_cache: bool = True):
        super(NCNLayer, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.cn_threshold = cn_threshold
        self.max_nodes_dense = max_nodes_dense
        self.cache_dir = cache_dir
        self.use_file_cache = use_file_cache
        
        self.node_transform = nn.Linear(in_feats, out_feats)
        self.neighbor_transform = nn.Linear(in_feats, out_feats)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)
        self.norm = LearnableZScoreNorm(out_feats)

        # ========== 缓存相关 ==========
        self._is_precomputed = False
        self._use_dense = False
        self._cache_device = None
        self._graph_hash = None
        
        # 稠密缓存（小图）
        self._attn_matrix_cache = None
        self._has_cn_dense_cache = None
        
        # 稀疏缓存（大图）
        self._cn_graph_cache = None
        self._attn_weights_cache = None
        self._has_cn_sparse_cache = None
    
    def _load_or_compute_cn(self, graph, device):
        """加载或计算 CN 数据（核心逻辑）"""
        graph_hash = _get_graph_hash(graph, self.cn_threshold)
        self._graph_hash = graph_hash
        
        # 1. 先查内存缓存
        if graph_hash in _CN_CACHE:
            print(f"[CN] 从内存加载: {graph_hash}")
            self._load_from_cn_data(_CN_CACHE[graph_hash], device)
            return
        
        # 2. 再查文件缓存
        if self.use_file_cache:
            cache_path = _get_cache_path(graph_hash, self.cache_dir)
            if os.path.exists(cache_path):
                print(f"[CN] 从文件加载: {cache_path}")
                cn_data = torch.load(cache_path, map_location='cpu')
                _CN_CACHE[graph_hash] = cn_data  # 存入内存缓存
                self._load_from_cn_data(cn_data, device)
                return
        
        # 3. 计算
        print(f"[CN] 计算中... (节点={graph.num_nodes()}, 边={graph.num_edges()})")
        cn_data = self._compute_cn_data(graph, device)
        
        # 4. 保存到内存缓存
        _CN_CACHE[graph_hash] = cn_data
        
        # 5. 保存到文件
        if self.use_file_cache and not cn_data.get('empty', False):
            cache_path = _get_cache_path(graph_hash, self.cache_dir)
            cn_data_cpu = self._cn_data_to_cpu(cn_data)
            torch.save(cn_data_cpu, cache_path)
            print(f"[CN] 已保存: {cache_path}")
        
        self._load_from_cn_data(cn_data, device)
    
    def _compute_cn_data(self, graph, device):
        """计算 CN 数据"""
        num_nodes = graph.num_nodes()
        
        if graph.num_edges() == 0:
            return {'empty': True}
        
        use_dense = num_nodes <= self.max_nodes_dense
        
        if use_dense:
            # 小图：稠密模式
            adj = graph.adjacency_matrix().to_dense().to(device).float()
            cn_matrix = torch.mm(adj, adj)
            cn_matrix.fill_diagonal_(0)
            cn_matrix = cn_matrix * (cn_matrix >= self.cn_threshold).float()
            
            has_cn = (cn_matrix.sum(dim=1, keepdim=True) > 0).float()
            
            attn_scores = torch.log1p(cn_matrix)
            attn_scores = attn_scores.masked_fill(cn_matrix == 0, -1e9)
            attn_matrix = F.softmax(attn_scores, dim=1)
            
            no_cn_mask = (has_cn.squeeze() == 0)
            attn_matrix[no_cn_mask] = 0
            
            print(f"[CN] 稠密模式: {num_nodes} 节点")
            
            return {
                'empty': False,
                'use_dense': True,
                'attn_matrix': attn_matrix,
                'has_cn': has_cn,
            }
        else:
            # 大图：稀疏模式
            src, dst = graph.edges()
            src_np, dst_np = src.cpu().numpy(), dst.cpu().numpy()
            
            data = np.ones(len(src_np), dtype=np.float32)
            adj = csr_matrix((data, (src_np, dst_np)), shape=(num_nodes, num_nodes))
            
            cn_sparse = adj @ adj
            cn_sparse.setdiag(0)
            cn_sparse.eliminate_zeros()
            
            cn_coo = cn_sparse.tocoo()
            mask = cn_coo.data >= self.cn_threshold
            rows, cols = cn_coo.row[mask], cn_coo.col[mask]
            cn_values = cn_coo.data[mask]
            
            if len(cn_values) == 0:
                return {'empty': True}
            
            cn_graph = dgl.graph((rows, cols), num_nodes=num_nodes, device=device)
            cn_weights = torch.tensor(cn_values, dtype=torch.float32, device=device)
            
            scores = torch.log1p(cn_weights)
            attn_weights = dgl.ops.edge_softmax(cn_graph, scores)
            has_cn = (cn_graph.in_degrees() > 0).float().unsqueeze(1)
            
            print(f"[CN] 稀疏模式: CN边数={cn_graph.num_edges()}")
            
            return {
                'empty': False,
                'use_dense': False,
                'cn_graph': cn_graph,
                'attn_weights': attn_weights,
                'has_cn': has_cn,
            }
    
    def _load_from_cn_data(self, cn_data, device):
        """从 cn_data 加载到实例变量"""
        if cn_data.get('empty', True):
            self._is_precomputed = True
            return
        
        self._use_dense = cn_data['use_dense']
        
        if self._use_dense:
            self._attn_matrix_cache = cn_data['attn_matrix'].to(device)
            self._has_cn_dense_cache = cn_data['has_cn'].to(device)
        else:
            self._cn_graph_cache = cn_data['cn_graph'].to(device)
            self._attn_weights_cache = cn_data['attn_weights'].to(device)
            self._has_cn_sparse_cache = cn_data['has_cn'].to(device)
        
        self._cache_device = device
        self._is_precomputed = True
    
    def _cn_data_to_cpu(self, cn_data):
        """将 cn_data 移到 CPU（用于保存文件）"""
        if cn_data.get('empty', True):
            return cn_data
        
        if cn_data['use_dense']:
            return {
                'empty': False,
                'use_dense': True,
                'attn_matrix': cn_data['attn_matrix'].cpu(),
                'has_cn': cn_data['has_cn'].cpu(),
            }
        else:
            return {
                'empty': False,
                'use_dense': False,
                'cn_graph': cn_data['cn_graph'].to('cpu'),
                'attn_weights': cn_data['attn_weights'].cpu(),
                'has_cn': cn_data['has_cn'].cpu(),
            }
    
    def _ensure_cache_device(self, device):
        """确保缓存在正确的设备上"""
        if self._cache_device == device:
            return
        
        if self._use_dense:
            if self._attn_matrix_cache is not None:
                self._attn_matrix_cache = self._attn_matrix_cache.to(device)
                self._has_cn_dense_cache = self._has_cn_dense_cache.to(device)
        else:
            if self._cn_graph_cache is not None:
                self._cn_graph_cache = self._cn_graph_cache.to(device)
                self._attn_weights_cache = self._attn_weights_cache.to(device)
                self._has_cn_sparse_cache = self._has_cn_sparse_cache.to(device)
        
        self._cache_device = device
    
    def forward(self, graph, feat):
        device = feat.device
        
        # ===== 懒加载：第一次 forward 自动加载/计算 =====
        if not self._is_precomputed:
            self._load_or_compute_cn(graph, device)
        
        node_out = self.node_transform(feat)
        
        # 使用预计算的缓存
        if self._is_precomputed:
            self._ensure_cache_device(device)
            if self._use_dense:
                cn_out = self._compute_cn_dense_cached(feat)
            else:
                cn_out = self._compute_cn_sparse_cached(feat)
        else:
            cn_out = self.neighbor_transform(feat)
        
        alpha = torch.sigmoid(self.alpha)
        output = alpha * node_out + (1 - alpha) * cn_out
        
        output = self.norm(output)
        output = self.activation(output)
        output = self.dropout(output)
        
        return output
    
    def _compute_cn_dense_cached(self, feat):
        """小图稠密缓存 forward"""
        h = self.neighbor_transform(feat)
        
        if self._attn_matrix_cache is None:
            return h
        
        out = torch.mm(self._attn_matrix_cache, h)
        out = self._has_cn_dense_cache * out + (1 - self._has_cn_dense_cache) * h
        
        return out
    
    def _compute_cn_sparse_cached(self, feat):
        """大图稀疏缓存 forward"""
        h = self.neighbor_transform(feat)
        
        if self._cn_graph_cache is None:
            return h
        
        cn_graph = self._cn_graph_cache
        
        with cn_graph.local_scope():
            cn_graph.ndata['h'] = h
            cn_graph.edata['attn'] = self._attn_weights_cache
            
            cn_graph.update_all(
                fn.u_mul_e('h', 'attn', 'm'),
                fn.sum('m', 'out')
            )
            
            out = cn_graph.ndata['out']
            out = self._has_cn_sparse_cache * out + (1 - self._has_cn_sparse_cache) * h
            
        return out
    
    def clear_cache(self):
        """清除实例缓存（不删除文件）"""
        self._is_precomputed = False
        self._use_dense = False
        self._attn_matrix_cache = None
        self._has_cn_dense_cache = None
        self._cn_graph_cache = None
        self._attn_weights_cache = None
        self._has_cn_sparse_cache = None


# ============ 工具函数 ============
def clear_all_cn_cache(cache_dir=".cn_cache"):
    """清除所有缓存（内存 + 文件）"""
    global _CN_CACHE, _MULTIHOP_CN_CACHE
    _CN_CACHE = {}
    _MULTIHOP_CN_CACHE = {}
    
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[CN] 已清除缓存目录: {cache_dir}")


class CombinedNCNLayer(nn.Module):
    """Combines standard GCN layer with NCN enhancement layer"""
    
    def __init__(self, gcn_layer, ncn_layer):
        super(CombinedNCNLayer, self).__init__()
        self.gcn_layer = gcn_layer
        self.ncn_layer = ncn_layer
        
        out_feats = ncn_layer.out_feats
        
        self.gate = nn.Sequential(
            nn.Linear(out_feats * 2, out_feats),
            nn.Sigmoid()
        )
    
    def forward(self, graph, feat):
        gcn_out = self.gcn_layer(graph, feat)
        ncn_out = self.ncn_layer(graph, feat)
        
        combined = torch.cat([gcn_out, ncn_out], dim=1)
        gate_weight = self.gate(combined)
        
        output = gate_weight * gcn_out + (1 - gate_weight) * ncn_out
        return output


class MoECombinedNCNLayer(nn.Module):
    """MoE-based combination of GCN and NCN layers
    
    将 GCN 和 NCN 视为两个专家，通过路由网络动态选择
    """
    
    def __init__(self, 
                 gcn_layer, 
                 ncn_layer,
                 in_feats: int,
                 top_k: int = 2,
                 noise_std: float = 0.1,
                 load_balance_weight: float = 0.01):
        super().__init__()
        self.gcn_layer = gcn_layer
        self.ncn_layer = ncn_layer
        self.num_experts = 2
        self.top_k = min(top_k, 2)
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight
        
        out_feats = ncn_layer.out_feats
        
        self.router = nn.Linear(in_feats, num_hops) 
        
        self.aux_loss = 0.0
        self._last_expert_stats = {}
    
    def forward(self, graph, feat):
        batch_size = feat.size(0)
        
        router_logits = self.router(feat)
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        if self.top_k < self.num_experts:
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
            router_probs = F.softmax(top_k_logits, dim=-1)
            
            sparse_probs = torch.zeros_like(router_logits)
            sparse_probs.scatter_(1, top_k_indices, router_probs)
            router_probs_full = sparse_probs
        else:
            router_probs_full = F.softmax(router_logits, dim=-1)
        
        gcn_out = self.gcn_layer(graph, feat)
        ncn_out = self.ncn_layer(graph, feat)
        
        expert_outputs = torch.stack([gcn_out, ncn_out], dim=1)
        
        weights = router_probs_full.unsqueeze(-1)
        output = (expert_outputs * weights).sum(dim=1)
        
        if self.training:
            self._compute_aux_loss(router_logits, router_probs_full)

        with torch.no_grad():
            self._last_expert_stats = {
                'gcn_avg_weight': router_probs_full[:, 0].mean().item(),
                'ncn_avg_weight': router_probs_full[:, 1].mean().item(),
                'gcn_selected_ratio': (router_probs_full[:, 0] > 0.5).float().mean().item(),
                'ncn_selected_ratio': (router_probs_full[:, 1] > 0.5).float().mean().item(),
            }
        return output
    
    def _compute_aux_loss(self, router_logits, router_probs):
        expert_usage = router_probs.mean(dim=0)
        router_prob_avg = F.softmax(router_logits, dim=-1).mean(dim=0)
        self.aux_loss = self.load_balance_weight * self.num_experts * (expert_usage * router_prob_avg).sum()
    
    def get_aux_loss(self):
        return self.aux_loss
    
    def get_expert_usage(self, graph, feat):
        with torch.no_grad():
            router_logits = self.router(feat)
            router_probs = F.softmax(router_logits, dim=-1)
            
            return {
                'gcn_avg_weight': router_probs[:, 0].mean().item(),
                'ncn_avg_weight': router_probs[:, 1].mean().item(),
                'gcn_selected_ratio': (router_probs[:, 0] > 0.5).float().mean().item(),
                'ncn_selected_ratio': (router_probs[:, 1] > 0.5).float().mean().item(),
            }


# ============================================================
# 新增：多跳 CN 预计算器
# ============================================================

class MultiHopCNComputer:
    """
    多跳 Common Neighbor 预计算器
    
    计算 1-hop, 2-hop, 3-hop 的 CN 注意力矩阵:
    - 1-hop: A^1 (直接邻居)
    - 2-hop: A^2 (传统 Common Neighbors)  
    - 3-hop: A^3 (3跳可达)
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
        
        # 缓存数据
        self._cn_data_list = None  # list of hop data
        self._is_precomputed = False
        self._use_dense = False
        self._cache_device = None
    
    def precompute(self, graph, device):
        """预计算所有跳数的 CN 矩阵"""
        if self._is_precomputed:
            self._ensure_device(device)
            return
        
        graph_hash = _get_multihop_hash(graph, self.max_hops, self.cn_threshold)
        
        # 1. 查内存缓存
        if graph_hash in _MULTIHOP_CN_CACHE:
            print(f"[MultiHop CN] 从内存加载: {graph_hash}")
            self._load_from_cache(_MULTIHOP_CN_CACHE[graph_hash], device)
            return
        
        # 2. 查文件缓存
        if self.use_file_cache:
            cache_path = os.path.join(self.cache_dir, f"multihop_cn_{graph_hash}.pt")
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(cache_path):
                print(f"[MultiHop CN] 从文件加载: {cache_path}")
                cn_data = torch.load(cache_path, map_location='cpu')
                _MULTIHOP_CN_CACHE[graph_hash] = cn_data
                self._load_from_cache(cn_data, device)
                return
        
        # 3. 计算
        print(f"[MultiHop CN] 计算中... (节点={graph.num_nodes()}, hops={self.max_hops})")
        cn_data = self._compute_all_hops(graph, device)
        
        # 4. 保存
        _MULTIHOP_CN_CACHE[graph_hash] = cn_data
        if self.use_file_cache:
            cache_path = os.path.join(self.cache_dir, f"multihop_cn_{graph_hash}.pt")
            torch.save(self._to_cpu(cn_data), cache_path)
            print(f"[MultiHop CN] 已保存: {cache_path}")
        
        self._load_from_cache(cn_data, device)
    
    def _compute_all_hops(self, graph, device):
        """计算所有跳数的 CN"""
        num_nodes = graph.num_nodes()
        use_dense = num_nodes <= self.max_nodes_dense
        
        if graph.num_edges() == 0:
            return {'empty': True, 'use_dense': use_dense, 'hop_data': []}
        
        if use_dense:
            return self._compute_dense(graph, device, num_nodes)
        else:
            return self._compute_sparse(graph, device, num_nodes)
    
    def _compute_dense(self, graph, device, num_nodes):
        """稠密模式"""
        adj = graph.adjacency_matrix().to_dense().to(device).float()
        
        hop_data = []
        adj_power = torch.eye(num_nodes, device=device)
        


        for hop in range(1, self.max_hops + 1):
            adj_power = torch.mm(adj_power, adj)
            
            cn_matrix = adj_power.clone()
            cn_matrix.fill_diagonal_(0)


            # ========== 新增：Top-K 稀疏化 ==========
            if hop >= 2:  # 2-hop 及以上才需要
                k = min(50, num_nodes - 1)  # 每个节点最多保留50个邻居
                
                # 找每行的 top-k
                top_values, top_indices = torch.topk(cn_matrix, k=k, dim=1)
                
                # 构建稀疏化后的矩阵
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
            print(f"  [Hop-{hop}] 有效节点: {int(has_cn.sum().item())}/{num_nodes}")
        
        return {'empty': False, 'use_dense': True, 'hop_data': hop_data}
    
    def _compute_sparse(self, graph, device, num_nodes):
        """稀疏模式"""
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

            # ========== 新增：Top-K 稀疏化 ==========
            if hop >= 2:
                k = 50
                cn_csr = cn_sparse.tocsr()
                
                # 对每行保留 top-k
                for i in range(num_nodes):
                    row_start, row_end = cn_csr.indptr[i], cn_csr.indptr[i+1]
                    if row_end - row_start > k:
                        row_data = cn_csr.data[row_start:row_end]
                        row_indices = cn_csr.indices[row_start:row_end]
                        
                        # 找 top-k
                        top_k_idx = np.argpartition(row_data, -k)[-k:]
                        
                        # 把非 top-k 的置零
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
                print(f"  [Hop-{hop}] 无有效 CN")
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
            print(f"  [Hop-{hop}] CN边数: {cn_graph.num_edges()}")
        
        return {'empty': False, 'use_dense': False, 'hop_data': hop_data}
    
    def _load_from_cache(self, cn_data, device):
        """从缓存加载"""
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
        """移到 CPU"""
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
        """确保正确设备"""
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
        用指定跳数的 CN 矩阵聚合特征
        
        Args:
            feat: [N, d] 节点特征
            hop_idx: 跳数索引 (0=1-hop, 1=2-hop, 2=3-hop)
            transform_fn: 特征变换函数
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
# 新增：稀疏多跳 MoE 层
# ============================================================

# class GatedFusion(nn.Module):
#     """节点级门控融合"""
#     def __init__(self, hidden_dim, dropout=0.1):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim * 3, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 2),
#             nn.Softmax(dim=-1)
#         )
    
#     def forward(self, gcn_out, moe_out, h_proj):
#         gate_input = torch.cat([gcn_out, moe_out, h_proj], dim=-1)
#         weights = self.gate(gate_input)  # [N, 2]
#         out = weights[:, 0:1] * gcn_out + weights[:, 1:2] * moe_out
#         return out
class GatedFusion(nn.Module):
    """节点级门控融合 - 三路融合"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.min_logits = nn.Parameter(torch.full((3,), init_min))
        self.temperature = nn.Parameter(torch.ones(1)) 
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # 改成3个权重
        )
        self._init_weights()
    
    def _init_weights(self):
        # ✅ 关键：初始化让三个分支权重接近均匀
        # 最后一层 bias 设为 0，weight 用小值
        nn.init.normal_(self.gate[-1].weight, std=0.001)
        nn.init.zeros_(self.gate[-1].bias)
    
    def forward(self, gcn_out, moe_out, h_proj):
        gate_input = torch.cat([gcn_out, moe_out, h_proj], dim=-1)
        logits = self.gate(gate_input)  # [N, 3]
        temp = F.softplus(self.temperature) + 0.01  # 确保 > 0
        weights = F.softmax(logits / temp, dim=-1)
        out = weights[:, 0:1] * gcn_out + weights[:, 1:2] * moe_out + weights[:, 2:3] * h_proj
        return out, weights

class SparseMultiHopMoE(nn.Module):
    """
    稀疏多跳 MoE 层
    
    架构:
    - 主路径: GCN (必选，不参与路由)
    - 增益路径: 1-hop, 2-hop, 3-hop CN (稀疏选择 top-k)
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
                 cache_dir: str = ".cn_cache"):
        super().__init__()
        
        self.gcn_layer = gcn_layer
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_hops = num_hops
        self.top_k = min(top_k, num_hops)
        self.noise_std = noise_std
        self.load_balance_weight = load_balance_weight
        
        # 多跳 CN 计算器
        self.cn_computer = MultiHopCNComputer(
            max_hops=num_hops,
            cn_threshold=cn_threshold,
            max_nodes_dense=max_nodes_dense,
            cache_dir=cache_dir,
        )
        
        # 每个跳数的变换层
        self.hop_transforms = nn.ModuleList([
            nn.Linear(in_feats, out_feats) for _ in range(num_hops)
        ])
        
        # # 路由器: 只在增益专家中选择
        # self.router = nn.Sequential(
        #     nn.Linear(in_feats, in_feats // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(in_feats // 2, num_hops)
        # )

        # 路由器：看 GCN输出 + 各hop输出
        self.router = nn.Sequential(
            nn.Linear(out_feats * (num_hops + 1), out_feats),  # 改这里！
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feats, num_hops)
        )
        # 整体增益系数
        self.alpha = nn.Parameter(torch.zeros(1))
        
        # 统计
        self.aux_loss = 0.0
        self._last_expert_stats = {}

        self.input_proj = nn.Linear(in_feats, out_feats) if in_feats != out_feats else nn.Identity()
        
        # 门控融合层
        # self.fusion = GatedFusion(out_feats, dropout=dropout)
        self.fusion = GatedFusion(out_feats, dropout=0.1)
        self.layer_norm = nn.LayerNorm(out_feats)
        # self.layer_norm = LearnableZScoreNorm(out_feats)
    def forward(self, graph, feat, main_out=None):
        device = feat.device

        h_original = feat
        # 预计算多跳 CN
        self.cn_computer.precompute(graph, device)
        
        # ===== 主路径 (必选) =====
        # main_out = self.gcn_layer(graph, feat)
        

        # ===== 主路径 =====
        if main_out is None:
            main_out = self.gcn_layer(graph, feat)  # 原有行为
        
        
        # ===== 计算所有跳数的增益 =====
        hop_gains = []
        for i in range(self.num_hops):
            gain = self.cn_computer.aggregate(
                feat, hop_idx=i, transform_fn=self.hop_transforms[i]
            )
            hop_gains.append(gain)
        
        # hop_gains = torch.stack(hop_gains, dim=1)  # [N, num_hops, out_feats]
        
        # # ===== 稀疏路由 =====
        # router_logits = self.router(feat)  # [N, num_hops]
        

        # ===== 关键：先拼接路由输入，再 stack =====
        router_input = torch.cat([main_out] + hop_gains, dim=-1)  # [N, out_feats * (num_hops+1)]
        router_logits = self.router(router_input)  # [N, num_hops]
        
        # 现在才 stack 用于后续加权
        hop_gains = torch.stack(hop_gains, dim=1)  # [N, num_hops, out_feats]
        
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Top-k 选择
        if self.top_k < self.num_hops:
            top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_logits, dim=-1)
            
            sparse_weights = torch.zeros_like(router_logits)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            router_probs = sparse_weights
        else:
            router_probs = F.softmax(router_logits, dim=-1)


        # # ===== 加这两行：混合均匀分布，防止极端 =====
        # uniform = torch.ones_like(router_probs) / self.num_hops
        # router_probs = 0.8 * router_probs + 0.2 * uniform  # 20%均匀 + 80%学习
        # # ============================================
        
        if self.training:
            router_probs = F.dropout(router_probs, p=0.1, training=True)
            # router_probs = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-8)
        # # ===== 加权增益 =====
        # weighted_gain = (hop_gains * router_probs.unsqueeze(-1)).sum(dim=1)
        
        # # ===== 融合: 主路径 + 增益 =====
        # alpha = torch.sigmoid(self.alpha)
        # output = main_out + alpha * weighted_gain
        

        # 加权聚合多跳增益
        # moe_out = (hop_gains * router_probs.unsqueeze(-1)).sum(dim=1)  # [N, out_feats]
        moe_out = main_out + (hop_gains * router_probs.unsqueeze(-1)).sum(dim=1)
        # 消融：Average Pooling 替代 MoE 路由
        # moe_out = main_out + hop_gains.mean(dim=1)  # 对 num_hops 维度取平均
        # ===== 门控融合 =====
        h_proj = self.input_proj(h_original)  # 投影原始特征到 out_feats
        # output = self.fusion(main_out, moe_out, h_proj)
        output, gate_weights = self.fusion(main_out, moe_out, h_proj)

        # # 残差 + LayerNorm
        # output = self.layer_norm(output + h_proj)
        # 残差 + LayerNorm
        output = self.layer_norm(output)


        # # 消融：去掉 GatedFusion，直接用 moe_out
        # h_proj = self.input_proj(h_original)
        # output = self.layer_norm(moe_out + h_proj)  # 只保留残差连接
        # gate_weights = torch.zeros(moe_out.size(0), 3, device=moe_out.device)  # 占位，避免统计报错
        # ===== 辅助损失 =====
        if self.training:
            self._compute_aux_loss(router_logits, router_probs)
        
        # 统计
        # with torch.no_grad():
        #     self._last_expert_stats = {'alpha': alpha.item()}
        #     for i in range(self.num_hops):
        #         self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()
        #         if self.top_k < self.num_hops:
        #             self._last_expert_stats[f'hop{i+1}_selected_ratio'] = (router_probs[:, i] > 0).float().mean().item()
        
        with torch.no_grad():
            # 记录门控权重

            self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
            # print(gate_weights[:, 0].mean().item())
            # print(gate_weights[:, 1].mean().item())
            self._last_expert_stats = {
                'gcn_weight': gate_weights[:, 0].mean().item(),
                'moe_weight': gate_weights[:, 1].mean().item(),
            }
            for i in range(self.num_hops):
                self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()
        
        return output
    
    def _compute_aux_loss(self, router_logits, router_probs):
        """负载均衡损失"""
        expert_usage = router_probs.mean(dim=0)
        router_prob_avg = F.softmax(router_logits, dim=-1).mean(dim=0)
        self.aux_loss = self.load_balance_weight * self.num_hops * (expert_usage * router_prob_avg).sum()

   
    def get_aux_loss(self):
        return self.aux_loss
    
    def get_expert_stats(self):
        return self._last_expert_stats

    def precompute_all(self, graph, feat):
        """
        在完整图上预计算所有节点的 CN 聚合结果（不应用 transform）
        
        Args:
            graph: 完整图
            feat: [N, in_feats] 完整图的原始特征
        """
        device = feat.device
        
        # 复用 cn_computer.precompute
        self.cn_computer.precompute(graph, device)
        
        # 预计算每个跳数的聚合结果（不应用 transform，避免计算图问题）
        self._precomputed_hop_feats = []
        with torch.no_grad():  # 关键：不创建计算图
            for i in range(self.num_hops):
                # 只做 CN 聚合，不应用 hop_transforms
                hop_feat = self.cn_computer.aggregate(feat, hop_idx=i, transform_fn=None)
                self._precomputed_hop_feats.append(hop_feat)
        
        print(f"[MoE] 预计算完成: {self.num_hops} 跳")

    # def forward_minibatch(self, feat, node_indices):
    #     """
    #     mini-batch forward：索引预计算的 CN 特征，然后应用 transform
        
    #     Args:
    #         feat: [batch_size, in_feats] 这个 batch 的原始特征
    #         node_indices: [batch_size] 节点在完整图中的索引
        
    #     Returns:
    #         [batch_size, out_feats]
    #     """
    #     h_original = feat
        
    #     # 索引预计算的 hop 特征（未 transform 的）
    #     hop_feats_raw = [
    #         self._precomputed_hop_feats[i][node_indices] 
    #         for i in range(self.num_hops)
    #     ]
        
    #     # 【关键】在这里应用 transform（可训练）
    #     hop_gains = [
    #         self.hop_transforms[i](hop_feats_raw[i])
    #         for i in range(self.num_hops)
    #     ]
        
    #     # 主路径
    #     main_out = self.input_proj(feat)
        
    #     # 路由
    #     router_input = torch.cat([main_out] + hop_gains, dim=-1)
    #     router_logits = self.router(router_input)
        
    #     if self.training and self.noise_std > 0:
    #         router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        
    #     # Top-k 选择
    #     if self.top_k < self.num_hops:
    #         top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
    #         top_k_weights = F.softmax(top_k_logits, dim=-1)
    #         sparse_weights = torch.zeros_like(router_logits)
    #         sparse_weights.scatter_(1, top_k_indices, top_k_weights)
    #         router_probs = sparse_weights
    #     else:
    #         router_probs = F.softmax(router_logits, dim=-1)
        
    #     if self.training:
    #         router_probs = F.dropout(router_probs, p=0.1, training=True)
        
    #     # 加权聚合
    #     hop_gains_stack = torch.stack(hop_gains, dim=1)
    #     moe_out = main_out + (hop_gains_stack * router_probs.unsqueeze(-1)).sum(dim=1)
        
    #     # 门控融合
    #     h_proj = self.input_proj(h_original)
    #     output, gate_weights = self.fusion(main_out, moe_out, h_proj)
    #     output = self.layer_norm(output + h_proj)
        
    #     # 辅助损失
    #     if self.training:
    #         self._compute_aux_loss(router_logits, router_probs)
        
    #     # 统计
    #     with torch.no_grad():
    #         self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
    #         self._last_expert_stats = {
    #             'gcn_weight': gate_weights[:, 0].mean().item(),
    #             'moe_weight': gate_weights[:, 1].mean().item(),
    #         }
    #         for i in range(self.num_hops):
    #             self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()
        
    #     return output


    def forward_minibatch(self, feat, node_indices, main_out=None, transformed_feat=None):
        """
        Mini-batch forward：索引预计算的 CN 特征
        
        Args:
            feat: [batch_size, in_feats] 这个 batch 的原始特征
            node_indices: [batch_size] 节点在完整图中的索引
            main_out: [batch_size, out_feats] 外部提供的主路径输出（如 DGA 的卷积）
        
        Returns:
            [batch_size, out_feats]
        """
        h_original = feat

        # ===== 主路径 =====
        if main_out is None:
            # 没有外部输入时，用投影（保持原行为）
            main_out = self.input_proj(feat)
        
        # ===== 索引预计算的 hop 特征，并应用 transform =====
        hop_feats_raw = [
            self._precomputed_hop_feats[i][node_indices] 
            for i in range(self.num_hops)
        ]
        hop_gains = [
            self.hop_transforms[i](hop_feats_raw[i])
            for i in range(self.num_hops)
        ]
        
        # ===== 路由（与 forward 一致）=====
        router_input = torch.cat([main_out] + hop_gains, dim=-1)
        router_logits = self.router(router_input)
        
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        
        # Top-k 选择
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
        
        # ===== 加权聚合（与 forward 一致）=====
        hop_gains_stack = torch.stack(hop_gains, dim=1)
        # moe_out = main_out + (hop_gains_stack * router_probs.unsqueeze(-1)).sum(dim=1)
        moe_out = (hop_gains_stack * router_probs.unsqueeze(-1)).sum(dim=1)
        # ===== 关键修改：使用外部传入的 transformed_feat =====
        if transformed_feat is not None:
            h_proj = transformed_feat
        else:
            # h_proj = self.input_proj(h_original)
            h_proj=h_original
 
        output, gate_weights = self.fusion(main_out, moe_out, h_proj)
        # output = self.layer_norm(output + h_proj)
        output = self.layer_norm(output)
        
        # ===== 辅助损失 =====
        if self.training:
            self._compute_aux_loss(router_logits, router_probs)
        
        # ===== 统计（与 forward 一致）=====
        with torch.no_grad():
            self._last_gate_weights = gate_weights.mean(dim=0).cpu().numpy()
            self._last_expert_stats = {
                'gcn_weight': gate_weights[:, 0].mean().item(),
                'moe_weight': gate_weights[:, 1].mean().item(),
            }
            for i in range(self.num_hops):
                self._last_expert_stats[f'hop{i+1}_avg_weight'] = router_probs[:, i].mean().item()
        
        return output
