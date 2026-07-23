"""Train a jixp-compiled MLP on MNIST.

1. compile the DSL:   python compiler.py examples/mnist/mnist.jixp
2. run this script:   uv run --with jax python examples/mnist/train.py

Downloads MNIST (~11 MB, four IDX files) into examples/mnist/data/mnist/ on the
first run. Dependency-free besides jax/numpy: hand-rolled cross-entropy and
SGD with momentum. Expect ~98% test accuracy in three epochs.
"""

import gzip
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mnist  # the generated module (lives next to this script)

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
DATA_DIR = Path(__file__).parent / "data" / "mnist"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def load(name):
    path = DATA_DIR / FILES[name]
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {FILES[name]} ...")
        urllib.request.urlretrieve(MIRROR + FILES[name], path)
    raw = gzip.decompress(path.read_bytes())
    if name.endswith("_x"):  # IDX image file: 16-byte header, then pixels
        pixels = np.frombuffer(raw, np.uint8, offset=16)
        return pixels.reshape(-1, 784).astype(np.float32) / 255.0
    return np.frombuffer(raw, np.uint8, offset=8).astype(np.int32)


train_x, train_y = load("train_x"), load("train_y")
test_x, test_y = jnp.array(load("test_x")), jnp.array(load("test_y"))

params = mnist.init_classifier(jax.random.PRNGKey(0))
momentum = jax.tree.map(jnp.zeros_like, params)


def loss_fn(params, x, y):
    logits = mnist.classifier(params, x)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -logp[jnp.arange(y.shape[0]), y].mean()


@jax.jit
def step(params, momentum, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    momentum = jax.tree.map(lambda m, g: 0.9 * m + g, momentum, grads)
    params = jax.tree.map(lambda p, m: p - 0.1 * m, params, momentum)
    return params, momentum, loss


@jax.jit
def accuracy(params, x, y):
    return (mnist.classifier(params, x).argmax(axis=-1) == y).mean()


BATCH = 128
rng = np.random.default_rng(0)
for epoch in range(3):
    order = rng.permutation(len(train_x))
    for i in range(0, len(order) - BATCH + 1, BATCH):
        batch = order[i : i + BATCH]
        params, momentum, loss = step(
            params, momentum, jnp.array(train_x[batch]), jnp.array(train_y[batch])
        )
    print(f"epoch {epoch}  loss {loss:.4f}  test acc {accuracy(params, test_x, test_y):.4f}")
