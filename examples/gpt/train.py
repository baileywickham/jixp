"""Train the jixp-compiled GPT on tiny-shakespeare, byte-level.

1. compile the DSL:   python compiler.py examples/gpt/gpt.jixp
2. run this script:   uv run --with jax python examples/gpt/train.py

Downloads tiny-shakespeare (~1.1 MB) into examples/gpt/data/ on the first run.
Dependency-free besides jax/numpy: vocab 256 means the bytes ARE the tokens
(no tokenizer), and Adam is hand-rolled with jax.tree.map. Trains on --seq
token windows (<= gpt.DIMS["block_size"]; the pos-embedding table is sliced
per sequence), then samples from the model.
"""

import argparse
import pickle
import time
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gpt  # the generated module (lives next to this script)

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).parent / "data"


def load_tokens():
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("downloading tinyshakespeare.txt ...")
        urllib.request.urlretrieve(URL, path)
    return np.frombuffer(path.read_bytes(), np.uint8).astype(np.int32)


def get_batch(rng, tokens, seq, batch):
    starts = rng.integers(0, len(tokens) - seq - 1, batch)
    x = np.stack([tokens[i : i + seq] for i in starts])
    y = np.stack([tokens[i + 1 : i + seq + 1] for i in starts])
    return jnp.array(x), jnp.array(y)


def loss_fn(params, x, y):
    logits = gpt.gpt(params, x)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.take_along_axis(logp, y[..., None], axis=-1).mean()


@jax.jit
def step(params, opt, x, y, lr):
    m, v, t = opt
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    t = t + 1
    m = jax.tree.map(lambda m_, g: 0.9 * m_ + 0.1 * g, m, grads)
    v = jax.tree.map(lambda v_, g: 0.999 * v_ + 0.001 * g * g, v, grads)
    scale = lr * jnp.sqrt(1 - 0.999**t) / (1 - 0.9**t)
    params = jax.tree.map(
        lambda p, m_, v_: p - scale * m_ / (jnp.sqrt(v_) + 1e-8), params, m, v
    )
    return params, (m, v, t), loss


def generate(params, key, prompt, n_new, seq, temperature):
    """Byte-level autoregressive sampling. The context is right-padded to a
    fixed length so the forward pass compiles once; causality keeps the
    padding out of the sampled position's logits."""
    ids = list(prompt.encode())
    for _ in range(n_new):
        key, sub = jax.random.split(key)
        ctx = jnp.array(ids[-seq:], dtype=jnp.int32)
        padded = jnp.zeros(seq, dtype=jnp.int32).at[: ctx.shape[0]].set(ctx)
        logits = gpt.gpt(params, padded[None])[0]
        nxt = jax.random.categorical(sub, logits[ctx.shape[0] - 1] / temperature)
        ids.append(int(nxt))
    return bytes(ids).decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample", type=int, default=400, help="bytes to sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default="ROMEO:")
    args = parser.parse_args()
    assert args.seq <= gpt.DIMS["block_size"], (
        f"--seq {args.seq} exceeds the compiled block size "
        f"{gpt.DIMS['block_size']} (edit gpt.jixp and recompile)"
    )

    tokens = load_tokens()
    split = int(len(tokens) * 0.9)
    train_tokens, val_tokens = tokens[:split], tokens[split:]

    params = gpt.init_gpt(jax.random.PRNGKey(0))
    zeros = jax.tree.map(jnp.zeros_like, params)
    opt = (zeros, zeros, jnp.int32(0))
    n_params = sum(a.size for a in jax.tree.leaves(params))
    print(f"{n_params:,} params, {len(train_tokens):,} train bytes")

    rng = np.random.default_rng(0)
    val_batch = get_batch(rng, val_tokens, args.seq, args.batch)
    t0 = time.perf_counter()
    for i in range(args.steps):
        x, y = get_batch(rng, train_tokens, args.seq, args.batch)
        params, opt, loss = step(params, opt, x, y, args.lr)
        if i % 200 == 0 or i == args.steps - 1:
            val = loss_fn(params, *val_batch)
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"step {i:5d}  train {loss:.3f}  val {val:.3f}  ({rate:.1f} steps/s)")

    with open(DATA_DIR / "params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(f"saved {DATA_DIR / 'params.pkl'}")

    print(generate(params, jax.random.PRNGKey(1), args.prompt,
                   args.sample, args.seq, args.temperature))


if __name__ == "__main__":
    main()
