# jisp

A tiny Lisp that compiles neural network definitions to pure-[JAX](https://github.com/jax-ml/jax) Python.

jisp is three small pieces, each built on the last:

1. **A parser** (`jisp.py`) — s-expressions to plain Python lists/tuples.
2. **An evaluator** (`jisp.py`) — a minimal Lisp: `quote`, `if`, `define`, `lambda`, arithmetic.
3. **A compiler** (`compiler.py`) — a JAX DSL: describe a model as s-expressions, get a
   dependency-free Python module of `init_*(key)` / `apply(params, x)` functions.
   The compiler evaluates dimension expressions with the interpreter from step 2.

```lisp
; examples/transformer.lisp
(let-dim
  (d 128) (heads 8) (layers 4))

(define attn_block (residual (chain (layernorm d) (attention d heads :causal))))
(define ffn_block (residual (chain (layernorm d) (mlp d [4d] d))))

(define encoder (repeat layers (chain attn_block ffn_block)))
```

```console
$ python compiler.py examples/transformer.lisp
wrote examples/transformer.py
```

The output is readable, framework-free JAX — every `define` becomes a pair of
functions, dims baked in as literals:

```python
def init_encoder(key):
    return [init_encoder_body(k) for k in jax.random.split(key, 4)]


def encoder(params, x):
    for layer_params in params:
        x = encoder_body(layer_params, x)
    return x
```

```python
import jax, transformer as t

params = t.init_encoder(jax.random.PRNGKey(0))
y = t.encoder(params, x)          # x: (batch, seq, 128)
```

## The DSL

Top-level forms:

| form | meaning |
|---|---|
| `(let-dim (name dim) ...)` | bind dimension names at compile time; later bindings may use earlier ones. Flat pairs `(let-dim d 128 heads 8)` also work |
| `(define name block)` | compile a block to `init_<name>(key)` and `<name>(params, x)` |

Blocks:

| block | meaning |
|---|---|
| `(mlp in [hidden...] out :activation gelu?)` | dense layers; activation (`relu` `gelu` `silu` `sigmoid` `tanh`, default `gelu`) after each hidden layer, never after the output layer |
| `(layernorm dim)` | layer normalization with scale and shift |
| `(attention dim heads :causal?)` | multi-head self-attention over `(..., seq, dim)`; `:causal` masks future positions |
| `(chain block...)` | sequential composition |
| `(residual block)` | `x + block(x)`; block must preserve dim |
| `(repeat count block)` | `count` copies with independent params; block must preserve dim |

A block position also accepts the name of an earlier `define` — references compile
to calls of that block's functions, and each use initializes fresh params (a
`define` is a template, not weight sharing).

### Dims

Anywhere a dimension is expected you can write a compile-time expression,
evaluated by the jisp interpreter:

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
- Required args are positional; options are keywords — valued (`:activation relu`)
  or flags (`:causal`).
- `define` names become Python function names, so they must be valid Python
  identifiers (`attn_block`, not `attn-block`).

## The Lisp

The interpreter that powers dim expressions also stands alone:

```console
$ python jisp.py program.lisp
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
$ uv run --with pytest -- pytest            # 106 tests
$ uv run --with pytest --with jax -- pytest # + integration: runs generated modules,
                                            #   checks shapes and causality
```

Design history lives in `docs/superpowers/`.
