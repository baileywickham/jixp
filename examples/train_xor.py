"""Train a jisp-compiled net on XOR — the full workflow in one file.

1. compile the DSL:   python compiler.py examples/xor.jisp
2. run this script:   uv run --with jax python examples/train_xor.py

The generated module (xor.py) is plain JAX, so jit/grad/tree_map all
apply directly; jisp itself is not needed at runtime.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent))
import xor  # the generated module

X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
Y = jnp.array([[0.0], [1.0], [1.0], [0.0]])

params = xor.init_net(jax.random.PRNGKey(0))


def loss_fn(params):
    pred = xor.net(params, X)
    return jnp.mean((pred - Y) ** 2)


@jax.jit
def step(params):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    return params, loss


for i in range(2001):
    params, loss = step(params)
    if i % 500 == 0:
        print(f"step {i:4d}  loss {loss:.6f}")

print("predictions:", [round(float(p), 3) for p in xor.net(params, X).ravel()])
