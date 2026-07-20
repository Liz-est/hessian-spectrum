"""
toy_model_C: single-layer vanilla decoder-only Transformer (方案 C).

Config: d=192, h=6, d_head=32, d_ff=1024, V=1024, context=128.
Post-LayerNorm, ReLU FFN, fixed sinusoidal PE, linear bias on, embed/head untied.
"""

from vanilla_model import ToyVanilla, ToyVanillaConfig

config_C = ToyVanillaConfig(
    vocab_size=1024,
    n_embd=192,
    n_head=6,
    head_dim=32,
    n_ffn=1024,
    n_layer=1,
    block_size=128,
)


def build():
    return ToyVanilla(config_C)


if __name__ == "__main__":
    model = build()
    bd = model.param_breakdown()
    total = bd["total"]

    def row(label, key):
        v = bd[key]
        print(f"  {label:<20}: {v:>9,} ({v/1e6:.3f}M, {100*v/total:.1f}%)")

    print(f"toy_model_C total params : {total:,} ({total/1e6:.3f}M)")
    row("Token Embedding", "token_embedding")
    row("LM Head", "lm_head")
    row("Embedding + LM Head", "embed+head")
    row("Self-attention", "self_attention")
    row("FFN", "ffn")
    row("LayerNorm", "layernorm")
    row("Transformer body", "transformer")
