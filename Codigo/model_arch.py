"""
Carga del modelo HANNAH 360M usando OLMo-core.
Replica exactamente la configuracion del repositorio.
"""

import sys
import types
import torch

# bettermap usa ForkProcess que no existe en Windows - mock necesario
mock = types.ModuleType('bettermap')
mock.bettermap = types.ModuleType('bettermap.bettermap')
sys.modules['bettermap'] = mock
sys.modules['bettermap.bettermap'] = mock.bettermap

from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.attention import AttentionBackendName

# Constantes de HANNAH 360M
VOCAB_SIZE = 32000
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 24


def build_model():
    """Construye el modelo HANNAH usando la misma config del repo."""
    config = TransformerConfig.olmo3_7B(
        vocab_size=VOCAB_SIZE,
        attn_backend=AttentionBackendName.torch
    )
    config.d_model = D_MODEL
    config.n_layers = N_LAYERS
    config.block.attention.n_heads = N_HEADS
    config.block.attention.n_kv_heads = N_HEADS
    config.block.feed_forward.hidden_size = int(D_MODEL * 8 / 3)

    return config.build()


def load_model(checkpoint_path, device='cpu'):
    """Carga el modelo desde un checkpoint .pt"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model()

    # Quitar prefijo _orig_mod. (agregado por torch.compile)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")
    print(f"Training step: {checkpoint.get('step', 'N/A')}")

    return model


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens=200, temperature=0.75,
             top_k=50, eos_token_id=None, device='cpu'):
    """Autoregressive generation (same loop as the repo)."""
    ids = input_ids.to(device)

    for _ in range(max_new_tokens):
        if ids.shape[1] > 1024:
            ids = ids[:, -1024:]

        logits = model(ids)
        logits = logits[:, -1, :] / temperature

        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        chosen = torch.multinomial(probs[0], 1)
        next_tok = top_idx[0][chosen]

        ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)

        if eos_token_id is not None and next_tok.item() == eos_token_id:
            break

    return ids
