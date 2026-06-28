import torch

from engine.config import ModelConfig
from engine.inference import greedy_decode
from engine.model import MiniQwenForCausalLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        hidden_act="silu",
        attention_bias=True,
        mlp_bias=False,
        tie_word_embeddings=False,
    )


def test_tiny_forward_shape():
    torch.manual_seed(0)
    model = MiniQwenForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    logits = model(input_ids)

    assert logits.shape == (1, 4, 32)


def test_tiny_greedy_decode_appends_tokens():
    torch.manual_seed(0)
    model = MiniQwenForCausalLM(tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    output_ids = greedy_decode(model, input_ids, max_new_tokens=2)

    assert output_ids.shape == (1, 5)
    assert torch.equal(output_ids[:, :3], input_ids)
