#!/usr/bin/env python3

import dgl
from dgl.data.utils import load_graphs
import torch

def check_dglgraph_format(filepath):
    try:
        print(f"正在检查文件: {filepath}")
        graph = load_graphs(filepath)[0][0]
        
        print("\n=== 基本信息 ===")
        print(f"节点数: {graph.num_nodes()}")
        print(f"边数: {graph.num_edges()}")
        print(f"节点数据字段: {list(graph.ndata.keys())}")
        print(f"边数据字段: {list(graph.edata.keys())}")
        
        print("\n=== 必需字段检查 ===")
        required_fields = ['feature', 'label', 'train_masks', 'val_masks', 'test_masks']
        missing_fields = []
        
        for field in required_fields:
            if field in graph.ndata:
                shape = graph.ndata[field].shape
                dtype = graph.ndata[field].dtype
                print(f"✅ {field}: shape={shape}, dtype={dtype}")
            else:
                print(f"❌ 缺少 {field}")
                missing_fields.append(field)
        
        print("\n=== 详细检查 ===")
        
        # 检查标签
        if 'label' in graph.ndata:
            labels = graph.ndata['label']
            unique_labels = torch.unique(labels)
            print(f"标签类型: {labels.dtype}")
            print(f"标签范围: {labels.min().item()} - {labels.max().item()}")
            print(f"唯一标签值: {unique_labels.tolist()}")
            print(f"标签分布: {torch.bincount(labels)}")
            
            # 检查是否是二分类
            if len(unique_labels) == 2 and set(unique_labels.tolist()) == {0, 1}:
                print("✅ 标签格式正确 (二分类: 0/1)")
            else:
                print("⚠️  标签格式可能需要调整 (GADBench期望二分类0/1)")
        
        # 检查特征
        if 'feature' in graph.ndata:
            features = graph.ndata['feature']
            print(f"特征维度: {features.shape}")
            print(f"特征类型: {features.dtype}")
            print(f"特征范围: {features.min().item():.4f} - {features.max().item():.4f}")
            
            # 检查异常值
            if torch.isnan(features).any():
                nan_count = torch.isnan(features).sum().item()
                print(f"⚠️  特征中包含 {nan_count} 个NaN值")
            else:
                print("✅ 特征无NaN值")
                
            if torch.isinf(features).any():
                inf_count = torch.isinf(features).sum().item()
                print(f"⚠️  特征中包含 {inf_count} 个无穷值")
            else:
                print("✅ 特征无无穷值")
        
        # 检查掩码格式
        mask_fields = ['train_masks', 'val_masks', 'test_masks']
        for mask_field in mask_fields:
            if mask_field in graph.ndata:
                mask = graph.ndata[mask_field]
                if len(mask.shape) == 2 and mask.shape[1] == 20:
                    print(f"✅ {mask_field} 格式正确: {mask.shape}")
                    # 检查每列的分布
                    for i in range(min(3, mask.shape[1])):  # 只检查前3列
                        count = mask[:, i].sum().item()
                        print(f"   列{i}: {count}个True节点")
                else:
                    print(f"❌ {mask_field} 格式错误: 期望[num_nodes, 20], 实际{mask.shape}")
        
        print(f"\n=== 总结 ===")
        if len(missing_fields) == 0:
            print("🎉 数据格式完整！可以直接改名使用！")
            print("\n📋 使用步骤:")
            print("1. mkdir -p datasets/eth_alphahomora/")
            print("2. cp datasets/eth_AlphaHomora.dglgraph datasets/eth_alphahomora/eth_alphahomora")
            print("3. python benchmark.py --trials 1 --datasets eth_alphahomora --models SpaceGNN")
            return True
        else:
            print(f"❌ 缺少 {len(missing_fields)} 个必需字段: {missing_fields}")
            print("🔧 需要处理后才能使用")
            return False
            
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False

if __name__ == "__main__":
    filepath = "datasets/eth_AlphaHomora.dglgraph"
    can_rename = check_dglgraph_format(filepath)
    
    print(f"\n{'='*50}")
    if can_rename:
        print("结论: ✅ 可以直接改名使用")
    else:
        print("结论: ❌ 需要先处理数据")