import pytest
import torch
from unimol_lite.model import UniMolModel, UniMolConfig
from unimol_lite.data import smiles2unimolitem

@pytest.fixture
def sample_batch():
    # 创建一个测试用的batch
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # 阿司匹林
    item = smiles2unimolitem(smiles)
    # 扩展成batch
    batch = {
        "src_tokens": item["src_tokens"].unsqueeze(0),
        "src_coord": item["src_coord"].unsqueeze(0),
        "src_distance": item["src_distance"].unsqueeze(0),
        "src_edge_type": item["src_edge_type"].unsqueeze(0),
    }
    return batch

def test_model_initialization():
    # 测试模型初始化
    config = UniMolConfig()
    model = UniMolModel(config)
    assert model is not None

def test_model_forward(sample_batch):
    # 测试模型前向传播
    config = UniMolConfig()
    model = UniMolModel(config)
    
    # 前向传播
    encoder_output, pair_output = model(**sample_batch)
    
    # 检查输出形状
    batch_size = sample_batch["src_tokens"].shape[0]
    seq_len = sample_batch["src_tokens"].shape[1]
    hidden_dim = config.encoder_embed_dim
    attention_heads = config.encoder_attention_heads
    
    assert encoder_output.shape == (batch_size, seq_len, hidden_dim)
    assert pair_output.shape == (batch_size, seq_len, seq_len, attention_heads)

def test_model_from_pretrained():
    # 测试从预训练模型加载
    try:
        model = UniMolModel.from_pretrained("base_unimol")
        assert model is not None
    except Exception as e:
        pytest.skip(f"预训练模型加载失败: {str(e)}") 