# jexprlisp

A tiny Lisp that compiles neural network definitions to pure-[JAX](https://github.com/jax-ml/jax) Python.
(The language and file extension are `jisp` — j-expressions.)

jexprlisp is three small pieces, each built on the last:

1. **A parser** (`jisp.py`) — s-expressions to plain Python lists/tuples.
2. **An evaluator** (`jisp.py`) — a minimal Lisp: `quote`, `if`, `define`, `lambda`, arithmetic.
3. **A compiler** (`compiler.py`) — a JAX DSL: describe a model as s-expressions, get a
   dependency-free Python module of `init_*(key)` / `apply(params, x)` functions.
   The compiler evaluates dimension expressions with the interpreter from step 2.

```lisp
; examples/transformer.jisp
(let-dim
  (d 128) (heads 8) (layers 4))

(define block (chain
  (residual (layernorm d)
            (attention d heads #:causal))
  (residual (layernorm d)
            (mlp d [4d] d))))

(define encoder (repeat layers block))
```

```console
$ python compiler.py examples/transformer.jisp
wrote examples/transformer.py
```

The output is readable, framework-free JAX — every `define` becomes a pair of
functions, dims baked in as literals:

```python
__all__ = [
    "init_block", "block",
    "init_encoder", "encoder",
]

def init_encoder(key):
    return [init_block(k) for k in jax.random.split(key, 4)]


@jax.jit
def encoder(params, x):
    for layer_params in params:
        x = block(layer_params, x)
    return x
```

```python
import jax, transformer as t

params = t.init_encoder(jax.random.PRNGKey(0))
y = t.encoder(params, x)          # x: (batch, seq, 128)
```

## Workflow

Write the `.jisp`, compile it, and import the generated module into ordinary
JAX code — jexprlisp is a build step, never a runtime dependency. A complete
train-XOR example lives in `examples/`:

```lisp
; examples/xor.jisp
(define net (mlp 2 [8 8] 1 #:activation tanh))
```

```python
# examples/train_xor.py (abridged)
import xor                                    # the generated module

params = xor.init_net(jax.random.PRNGKey(0))

def loss_fn(params):
    return jnp.mean((xor.net(params, X) - Y) ** 2)

@jax.jit
def step(params):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return jax.tree.map(lambda p, g: p - 0.1 * g, params, grads), loss
```

```console
$ python compiler.py examples/xor.jisp
$ uv run --with jax python examples/train_xor.py
step    0  loss 1.472630
step 2000  loss 0.000000
predictions: [0.0, 1.0, 1.0, 0.0]
```

Params are ordinary pytrees and the generated functions are pure, so `grad`,
`vmap`, optax optimizers, and orbax checkpointing all apply directly. Every
public apply function ships wrapped in `@jax.jit` (internals are inlined at
trace time); use `jax.disable_jit()` when you want to step through eagerly.

## The DSL

Top-level forms:

| form | meaning |
|---|---|
| `(let-dim (name dim) ...)` | bind dimension names at compile time; later bindings may use earlier ones. Flat pairs `(let-dim d 128 heads 8)` also work |
| `(define name block)` | compile a block to `init_<name>(key)` and `<name>(params, x)` |

Blocks:

| block | meaning |
|---|---|
| `(mlp in [hidden...] out #:activation gelu? #:bias?)` | dense layers; activation (`relu` `gelu` `silu` `sigmoid` `tanh`, default `gelu`) after each hidden layer, never after the output layer; bias-free unless `#:bias` |
| `(layernorm dim)` | layer normalization with scale and shift |
| `(embedding vocab dim)` | lookup table: int token ids `(..., seq)` → `(..., seq, dim)`; only valid at the start of a model |
| `(attention dim heads #:causal?)` | multi-head self-attention over `(..., seq, dim)`; `#:causal` masks future positions |
| `(chain block...)` | sequential composition |
| `(residual block...)` | `x + body(x)`; body must preserve dim |
| `(repeat count block...)` | `count` copies with independent params; body must preserve dim |

Multi-block `residual`/`repeat` bodies chain implicitly: `(residual a b)`
is `(residual (chain a b))`.

A block position also accepts the name of an earlier `define` — references compile
to calls of that block's functions, and each use initializes fresh params (a
`define` is a template, not weight sharing).

### Dims

Anywhere a dimension is expected you can write a compile-time expression,
evaluated by the jexprlisp interpreter:

```lisp
(let-dim (d 128) (ffn (* 4 d)))
(mlp d [4d (/ d 2)] d)        ; 4d is sugar for (* 4 d)
```

Dims must resolve to positive integers at compile time, which buys static shape
checking: `chain` verifies adjacent blocks agree, `residual`/`repeat` require
dim-preserving bodies, and `attention` requires `heads` to divide `dim` — all
before any Python runs.

### Syntax notes

- `(...)` is always a form; `[...]` is always data (a tuple), e.g. mlp hidden dims.
- Required args are positional; options are keywords — valued (`#:activation relu`)
  or flags (`#:causal`).
- `;` starts a line comment.
- Names may use Racket-style hyphens (`attn-block`); they become underscores
  (`attn_block`) in the generated Python.

## The Lisp

The interpreter that powers dim expressions also stands alone:

```console
$ python jisp.py program.jisp
```

```lisp
(define fact (lambda (n) (if (< n 2) 1 (* n (fact (- n 1))))))
(fact 5)                       ; prints 120
```

Atoms are ints, floats, and symbols; special forms are `quote`, `if`, `define`,
and `lambda`; builtins are `+ - * / < > <= >= =`. Parse errors and eval errors
exit 1 with a message on stderr.

## Development

No runtime dependencies. Tests need pytest; the generated-code integration
tests additionally use JAX and skip cleanly without it:

```console
$ uv run --with pytest -- pytest            # unit tests
$ uv run --with pytest --with jax -- pytest # + integration: runs generated modules,
                                            #   checks shapes and causality
```
