"""
toy_model_A: single-layer LLaMA-style decoder (方案 A).

Config: d=160, h=5, d_head=32, d_ff=448, V=1024, embed/head untied.
"""

from model import ToyLlama, ToyLlamaConfig

config_A = ToyLlamaConfig(
    vocab_size=1024,
    n_embd=160,
    n_head=5,
    head_dim=32,
    n_ffn=448,
    n_layer=1,
    block_size=512,
)


def build():
    return ToyLlama(config_A)


if __name__ == "__main__":
    model = build()
    bd = model.param_breakdown()
    total = bd["total"]
    print(f"toy_model_A total params : {total:,} ({total/1e6:.3f}M)")
    print(f"  embed + LM head        : {bd['embed+head']:,} "
          f"({bd['embed+head']/1e6:.3f}M, {100*bd['embed+head']/total:.1f}%)")
    print(f"  transformer body       : {bd['transformer']:,} "
          f"({bd['transformer']/1e6:.3f}M, {100*bd['transformer']/total:.1f}%)")
