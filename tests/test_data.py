import pytest
import torch
from unimol_lite.data import smiles2unimolitem

def test_smiles2unimolitem():
    # 测试一个简单的分子
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    item = smiles2unimolitem(smiles)
    
    # 检查返回的item是否包含所有必需的键
    assert "src_tokens" in item
    assert "src_coord" in item
    assert "src_distance" in item
    assert "src_edge_type" in item
    
    # 检查张量的形状
    num_atoms = len(smiles.split("C")) - 1  # 粗略估计原子数
    assert item["src_tokens"].shape[0] > 0
    assert item["src_coord"].shape == (item["src_tokens"].shape[0], 3)
    assert item["src_distance"].shape == (item["src_tokens"].shape[0], item["src_tokens"].shape[0])
    assert item["src_edge_type"].shape == (item["src_tokens"].shape[0], item["src_tokens"].shape[0])
    
    # 检查数据类型
    assert isinstance(item["src_tokens"], torch.Tensor)
    assert isinstance(item["src_coord"], torch.Tensor)
    assert isinstance(item["src_distance"], torch.Tensor)
    assert isinstance(item["src_edge_type"], torch.Tensor)

def test_smiles2unimolitem_invalid_smiles():
    # 测试无效的SMILES
    with pytest.raises(Exception):
        smiles2unimolitem("invalid_smiles") 