"""
llama_B: single-layer LLaMA-style decoder (方案 B).

Config: d=256, h=4, d_head=64, d_ff=680, V=1024, embed/head untied.
"""

from llama_model import ToyLlama, ToyLlamaConfig

config_B = ToyLlamaConfig(
    vocab_size=1024,
    n_embd=256,
    n_head=4,
    head_dim=64,
    n_ffn=680,
    n_layer=1,
    block_size=512,
)


def build():
    return ToyLlama(config_B)


if __name__ == "__main__":
    model = build()
    bd = model.param_breakdown()
    total = bd["total"]
    print(f"llama_B total params : {total:,} ({total/1e6:.3f}M)")
    print(f"  embed + LM head        : {bd['embed+head']:,} "
          f"({bd['embed+head']/1e6:.3f}M, {100*bd['embed+head']/total:.1f}%)")
    print(f"  transformer body       : {bd['transformer']:,} "
          f"({bd['transformer']/1e6:.3f}M, {100*bd['transformer']/total:.1f}%)")
