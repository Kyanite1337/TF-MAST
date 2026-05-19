import pytest
import torch

from tfmast.config import load_config
from tfmast.models.heads import build_head
from tfmast.models.mae import MaskedAutoencoder
from tfmast.models.swin_emg import SwinEMGEncoder
from tfmast.models.tfc import TFCModel, _nt_xent
from tfmast.training.masks import MaskBank


def test_mask_bank_produces_expected_shape_and_sensor_masks():
    x = torch.randn(8, 16, 40)
    masks = MaskBank(mask_ratio=0.5, strategies=["sensor"], patch_size=(1, 4))(x)

    assert masks.encoder_mask.shape == (8, 16, 10)
    assert masks.decoder_mask.shape == (8, 16, 10)
    assert masks.encoder_mask.any()
    per_channel = masks.encoder_mask.flatten(2).all(dim=-1)
    assert per_channel.any()


def test_swin_encoder_returns_embedding_tokens_and_bypass():
    cfg = load_config(overrides={"model.embed_dim": 32, "model.depths": [1, 1], "model.num_heads": [2, 4]})
    encoder = SwinEMGEncoder.from_config(cfg)
    x = torch.randn(4, 16, 40)

    pooled, tokens, bypass = encoder(x, return_tokens=True, return_bypass=True)

    assert pooled.shape == (4, 32)
    assert tokens.ndim == 3
    assert bypass.shape == (4, 32)


def test_mae_forward_reconstructs_signal_and_returns_masked_loss():
    cfg = load_config(overrides={"model.embed_dim": 32, "model.depths": [1, 1], "model.num_heads": [2, 4]})
    model = MaskedAutoencoder.from_config(cfg)
    x = torch.randn(2, 16, 40)

    out = model(x)

    assert out.reconstruction.shape == x.shape
    assert out.loss.ndim == 0
    assert out.loss.requires_grad


def test_tfc_model_outputs_all_loss_terms():
    cfg = load_config(overrides={"model.embed_dim": 32, "model.depths": [1, 1], "model.num_heads": [2, 4]})
    model = TFCModel.from_config(cfg)
    x = torch.randn(4, 16, 40)

    out = model(x)

    assert set(out.losses) == {"loss", "loss_time", "loss_freq", "loss_consistency", "embedding_similarity"}
    assert out.losses["loss"].requires_grad


def test_nt_xent_accepts_half_precision_embeddings_without_mask_overflow():
    z1 = torch.randn(4, 32, dtype=torch.float16)
    z2 = torch.randn(4, 32, dtype=torch.float16)

    loss = _nt_xent(z1, z2, temperature=0.2)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_classifier_heads_forward_for_mlp_and_skip_mamba_when_unavailable():
    tokens = torch.randn(3, 20, 32)
    pooled = torch.randn(3, 32)
    bypass = torch.randn(3, 32)

    mlp = build_head("mlp", embed_dim=32, num_classes=53, bypass=True)
    assert mlp(tokens=tokens, pooled=pooled, bypass=bypass).shape == (3, 53)

    pytest.importorskip("mamba_ssm")
    mamba = build_head("mamba", embed_dim=32, num_classes=53, bypass=True)
    bimamba = build_head("bimamba", embed_dim=32, num_classes=53, bypass=True)
    assert mamba(tokens=tokens, pooled=pooled, bypass=bypass).shape == (3, 53)
    assert bimamba(tokens=tokens, pooled=pooled, bypass=bypass).shape == (3, 53)
