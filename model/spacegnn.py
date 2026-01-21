"""
SpaceGNN: Multi-Geometric Graph Neural Network for Graph Anomaly Detection

This module implements SpaceGNN, a graph neural network that operates simultaneously
in three geometric spaces (hyperbolic, spherical, and Euclidean) to capture complex
graph structures for anomaly detection tasks.

Paper Reference:
    SpaceGNN: Multi-Space Graph Neural Network for Graph Anomaly Detection
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


# ============================================================
# Geometric Operations
# ============================================================

def expmap(e, c):
    """Exponential map for geometric transformations"""
    sqrt_c = abs(c) ** 0.5
    e_norm = torch.clamp_min(e.norm(dim=-1, p=2, keepdim=True), 1e-15)
    if float(c) < 0:
        hemb = torch.tanh(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    else:
        hemb = torch.tan(sqrt_c * e_norm) * e / (sqrt_c * e_norm)
    return hemb


def logmap(h, c):
    """Logarithmic map for geometric transformations"""
    sqrt_c = abs(c) ** 0.5
    h_norm = h.norm(dim=-1, p=2, keepdim=True).clamp_min(1e-15)
    if float(c) < 0:
        scale = 1. / sqrt_c * torch.arctanh(sqrt_c * h_norm) / h_norm
    else:
        scale = 1. / sqrt_c * torch.atan(sqrt_c * h_norm) / h_norm
    return scale * h


def proj(x, c):
    """Projection to constrain embeddings within valid range"""
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-15)
    maxnorm = (1 - 1e-5) / (abs(c) ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


# ============================================================
# Neural Network Layers
# ============================================================

class CustomLinear(nn.Linear):
    """Linear layer with custom initialization"""
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)


class LinearLayer(nn.Module):
    """Two-layer MLP with normalization and dropout"""
    def __init__(self, in_dim, hid_dim, out_dim, drop_rate, final=True):
        super(LinearLayer, self).__init__()
        self.linear1 = CustomLinear(in_dim, hid_dim)
        self.linear2 = CustomLinear(hid_dim, out_dim)
        self.act = nn.ELU()
        self.drop = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(hid_dim)
        self.final = final

    def forward(self, h):
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.norm(h)
        h = self.linear2(h)
        if self.final:
            h = self.act(h)
            h = self.drop(h)
        return h


class CurvLayer(nn.Module):
    """Graph convolution layer with curvature-aware message passing"""
    def __init__(self, in_dim, hid_dim, out_dim, drop_rate):
        super(CurvLayer, self).__init__()
        self.edge_linear = LinearLayer(in_dim * 2, hid_dim, hid_dim, drop_rate)
        self.out_linear = CustomLinear(hid_dim, out_dim)
        self.edge_bn = nn.BatchNorm1d(hid_dim)

    def edge_udf(self, c):
        """Create edge message function with curvature c"""
        edge_linear = self.edge_linear
        def weighted_message(edges):
            src = edges.src['h']
            dst = edges.dst['h']

            if float(c) == 0:
                coef = 1 - torch.sigmoid((src - dst).norm(dim=-1, p=2, keepdim=True))
            else:
                dist = (src - dst).norm(dim=-1, p=2, keepdim=True)
                multi = (src * dst).sum(dim=-1, keepdim=True)
                coef =  1 - torch.sigmoid(2 * dist - 2 * c * ((dist ** 3) / 3 + multi * (dist ** 2)))

            msg = torch.cat([coef * src + src, dst], dim=-1)
            msg = edge_linear(msg)
            return {'msg': msg}
        return weighted_message

    def forward(self, graph, features, c):
        with graph.local_scope():
            if not (float(c) == 0):
                features = expmap(features, c)
                features = proj(features, c)
                features = logmap(features, c)

            src_feats = features
            dst_feats = features[:graph.num_dst_nodes()]
            graph.srcdata['h'] = src_feats
            graph.dstdata['h'] = dst_feats

            graph.apply_edges(self.edge_udf(c))

            graph.edata['msg'] = self.edge_bn(graph.edata['msg'])
            graph.update_all(fn.copy_e('msg', 'msg'), fn.sum('msg', 'out'))
            out = graph.dstdata.pop('out')
            out = self.out_linear(out)
            if not (float(c) == 0):
                out = expmap(out, c)
                out = proj(out, c)
                out = logmap(out, c)
                out = F.selu(out)
            out += dst_feats

            return out


class CurvGNN(nn.Module):
    """Multi-layer curvature-aware GNN"""
    def __init__(self, in_dim, hid_dim, out_dim, layer_num, drop_rate, moe_layer=None):
        super(CurvGNN, self).__init__()
        self.in_linears = nn.ModuleList()
        self.gnns = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.out_linears = nn.ModuleList()
        self.layer_num = layer_num
        self.moe_layer = moe_layer
        for i in range(layer_num):
            self.in_linears.append(LinearLayer(in_dim, hid_dim, hid_dim, drop_rate))
            self.gnns.append(CurvLayer(hid_dim, hid_dim, hid_dim, drop_rate))
            self.bns.append(nn.BatchNorm1d(hid_dim))
            self.out_linears.append(CustomLinear(hid_dim * 2, hid_dim))

        self.out_linear = CustomLinear(hid_dim * layer_num, out_dim)
        self.act = F.selu

    def forward(self, blocks, c, node_indices=None):
        final_num = blocks[0].num_dst_nodes()
        results = []

        for i in range(self.layer_num):
            inter_results = []
            h = self.in_linears[i](blocks[0].srcdata['feature_{}'.format(i)])
            transformed_feat = h[:final_num]  # Save transformed features
            inter_results.append(h[:final_num])
            h = self.gnns[i](blocks[0], h, c[i])
            h = self.bns[i](h)
            h = self.act(h)
            # ===== MoE fusion only in first layer =====
            if i == 0 and self.moe_layer is not None and node_indices is not None:
                h_moe = self.moe_layer.forward_minibatch(
                    feat=transformed_feat,
                    node_indices=node_indices,
                    main_out=h[:final_num]
                )
                h = h_moe
            # ==========================================
            inter_results.append(h[:final_num])
            h = torch.stack(inter_results, dim=1)
            h = h.reshape(h.shape[0], -1)
            h = self.out_linears[i](h)
            results.append(h)

        h = torch.stack(results, dim=1)
        h = h.reshape(h.shape[0], -1)
        h = self.out_linear(h)

        return h.log_softmax(dim=-1)


class SpaceGNNOriginal(nn.Module):
    """
    SpaceGNN that operates in three geometric spaces

    Uses blocks and batch training for efficient processing.
    Supports MoE layer integration for enhanced performance.
    """
    def __init__(self, in_dim, hid_dim, out_dim, layer_num, drop_rate, cneg, cpos, moe_layer=None):
        super(SpaceGNNOriginal, self).__init__()
        self.curvgnnneg = CurvGNN(in_dim, hid_dim, out_dim, layer_num, drop_rate, moe_layer=moe_layer)
        self.curvgnnpos = CurvGNN(in_dim, hid_dim, out_dim, layer_num, drop_rate, moe_layer=moe_layer)
        self.eucgnn = CurvGNN(in_dim, hid_dim, out_dim, layer_num, drop_rate, moe_layer=moe_layer)
        self.cneg = nn.Parameter(cneg)
        self.cpos = nn.Parameter(cpos)
        self.zeros = [0] * layer_num
        # Save reference for easy aux_loss access
        self.moe_layer = moe_layer

    def forward(self, blocks, node_indices=None):
        h1 = self.curvgnnneg(blocks, self.cneg, node_indices=node_indices)
        h2 = self.curvgnnpos(blocks, self.cpos, node_indices=node_indices)
        h3 = self.eucgnn(blocks, self.zeros, node_indices=node_indices)

        return h1, h2, h3

    def get_aux_loss(self):
        """Get MoE auxiliary loss"""
        if self.moe_layer is not None:
            return self.moe_layer.get_aux_loss()
        return 0.0
