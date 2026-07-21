import importlib.util
import subprocess
import sys

import pytest

from compiler import CompileError, compile_source

MLP_SRC = "(define model (mlp 784 [256 256] 10))"


def test_mlp_emits_init_and_apply():
    out = compile_source(MLP_SRC)
    assert "def init_model(key):" in out
    assert "def model(params, x):" in out
    assert "init_dense(keys[0], 784, 256)" in out
    assert "init_dense(keys[1], 256, 256)" in out
    assert "init_dense(keys[2], 256, 10)" in out


def test_generated_source_is_valid_python():
    compile(compile_source(MLP_SRC), "<generated>", "exec")


def test_mlp_with_no_hidden_layers():
    out = compile_source("(define lin (mlp 8 [] 4))")
    assert "jax.random.split(key, 1)" in out
    assert "init_dense(keys[0], 8, 4)" in out


def test_layernorm_emits_scale_and_shift():
    out = compile_source("(define norm (layernorm 128))")
    assert "def init_norm(key):" in out
    assert "jnp.ones(128), jnp.zeros(128)" in out
    assert "init_dense" not in out


def test_chain_emits_sublayers_and_threads_x():
    out = compile_source("(define model (chain (mlp 784 [256] 128) (layernorm 128)))")
    assert "def model_layer0(params, x):" in out
    assert "def model_layer1(params, x):" in out
    assert "x = model_layer0(params[0], x)" in out
    assert "x = model_layer1(params[1], x)" in out
    compile(out, "<generated>", "exec")


def test_mlp_defaults_to_gelu():
    out = compile_source(MLP_SRC)
    assert "jax.nn.gelu(x @ W + b)" in out
    assert "relu" not in out


def test_mlp_explicit_activation():
    out = compile_source("(define m (mlp 8 [16] 4 #:activation relu))")
    assert "jax.nn.relu(x @ W + b)" in out


def test_mlp_tanh_uses_jnp():
    out = compile_source("(define m (mlp 8 [16] 4 #:activation tanh))")
    assert "jnp.tanh(x @ W + b)" in out


def test_mlp_unknown_activation_raises():
    with pytest.raises(CompileError, match="unknown activation 'elu'"):
        compile_source("(define m (mlp 8 [16] 4 #:activation elu))")


def test_mlp_bare_activation_gives_migration_error():
    with pytest.raises(CompileError, match="#:activation relu"):
        compile_source("(define m (mlp 8 [16] 4 relu))")


def test_clojure_style_keyword_gives_migration_error():
    with pytest.raises(CompileError, match="Racket keyword syntax, e.g. #:causal"):
        compile_source("(define attn (attention 32 4 :causal))")


def test_line_comments_are_ignored():
    src = """
    ; a whole-line comment
    (define net (mlp 2 [8] 1))  ; trailing comment
    """
    out = compile_source(src)
    assert "def init_net(key):" in out


def test_hyphenated_names_mangle_to_underscores():
    src = """
    (let-dim (d 16))
    (define attn-block (residual (layernorm d)))
    (define stack (repeat 2 attn-block))
    """
    out = compile_source(src)
    assert "def attn_block(params, x):" in out
    assert "[init_attn_block(k) for k in jax.random.split(key, 2)]" in out


def test_mangled_name_collision_raises():
    with pytest.raises(CompileError, match="collides"):
        compile_source("(define a-b (layernorm 4)) (define a_b (layernorm 4))")


def test_let_dim_bracket_bindings():
    out = compile_source("(let-dim [d 16] [heads 4]) (define m (layernorm (* d heads)))")
    assert "jnp.ones(64), jnp.zeros(64)" in out


def test_option_value_missing_raises():
    with pytest.raises(CompileError, match="#:activation expects a value"):
        compile_source("(define m (mlp 8 [16] 4 #:activation))")


def test_duplicate_option_raises():
    with pytest.raises(CompileError, match="duplicate mlp option"):
        compile_source("(define m (mlp 8 [16] 4 #:activation relu #:activation tanh))")


def test_positional_after_keyword_raises():
    with pytest.raises(CompileError, match="positional arg .* after keyword"):
        compile_source("(define attn (attention 32 #:causal 4))")


def test_unknown_option_lists_valid_ones():
    with pytest.raises(CompileError, match="unknown mlp option #:casual.*#:activation"):
        compile_source("(define m (mlp 8 [16] 4 #:casual))")


def test_chain_dimension_mismatch_raises():
    with pytest.raises(CompileError, match="mismatch"):
        compile_source("(define m (chain (mlp 8 [] 4) (layernorm 5)))")


def test_top_level_must_be_define():
    with pytest.raises(CompileError, match="top-level"):
        compile_source("(mlp 1 () 2)")


def test_unknown_block_raises():
    with pytest.raises(CompileError, match="unknown block"):
        compile_source("(define m (conv 3 3))")


def test_bad_mlp_args_raise():
    with pytest.raises(CompileError, match="mlp expects"):
        compile_source("(define m (mlp 784 256 10))")


def test_duplicate_define_raises():
    with pytest.raises(CompileError, match="duplicate"):
        compile_source("(define m (layernorm 4)) (define m (layernorm 4))")


def test_empty_source_raises():
    with pytest.raises(CompileError, match="no forms"):
        compile_source("")


def test_let_dim_binds_name_to_dim():
    out = compile_source("(let-dim d 1024) (define m (mlp d [256] d))")
    assert "init_dense(keys[0], 1024, 256)" in out
    assert "init_dense(keys[1], 256, 1024)" in out
    assert "let" not in out


def test_let_dim_works_in_hidden_dims_and_layernorm():
    out = compile_source(
        "(let-dim h 64) (define m (chain (mlp 8 [h h] h) (layernorm h)))"
    )
    assert "init_dense(keys[0], 8, 64)" in out
    assert "init_dense(keys[2], 64, 64)" in out
    assert "jnp.ones(64), jnp.zeros(64)" in out


def test_let_dim_can_reference_earlier_let_dim():
    out = compile_source("(let-dim d 32) (let-dim e d) (define m (layernorm e))")
    assert "jnp.ones(32), jnp.zeros(32)" in out


def test_let_dim_participates_in_chain_dim_check():
    with pytest.raises(CompileError, match="mismatch"):
        compile_source(
            "(let-dim d 4) (define m (chain (mlp 8 [] d) (layernorm 5)))"
        )


def test_unbound_dim_raises():
    with pytest.raises(CompileError, match="unbound symbol 'd'"):
        compile_source("(define m (mlp d [] 4))")


def test_duplicate_let_dim_raises():
    with pytest.raises(CompileError, match="duplicate let-dim"):
        compile_source("(let-dim d 4) (let-dim d 8) (define m (layernorm d))")


def test_bad_let_dim_value_raises():
    with pytest.raises(CompileError, match="bad dim"):
        compile_source("(let-dim d (4 5)) (define m (layernorm d))")


def test_paren_hidden_dims_give_migration_error():
    with pytest.raises(CompileError, match=r"bracket tuple like \[256 256\]"):
        compile_source("(define m (mlp 784 (256 256) 10))")


def test_dim_arithmetic_expression():
    out = compile_source("(let-dim d 32) (define m (mlp d [(* 2 d)] (- d 24)))")
    assert "init_dense(keys[0], 32, 64)" in out
    assert "init_dense(keys[1], 64, 8)" in out


def test_let_dim_expression_value():
    out = compile_source("(let-dim d 32) (let-dim ffn (* 4 d)) (define m (layernorm ffn))")
    assert "jnp.ones(128), jnp.zeros(128)" in out


def test_coefficient_sugar():
    out = compile_source("(let-dim d 32) (define m (mlp d [4d] d))")
    assert "init_dense(keys[0], 32, 128)" in out


def test_coefficient_sugar_inside_expression():
    out = compile_source("(let-dim d 16) (define m (layernorm (+ 2d 1d)))")
    assert "jnp.ones(48), jnp.zeros(48)" in out


def test_integral_division_dim():
    out = compile_source("(let-dim d 32) (define m (layernorm (/ d 4)))")
    assert "jnp.ones(8), jnp.zeros(8)" in out


def test_non_integral_dim_raises():
    with pytest.raises(CompileError, match="positive int"):
        compile_source("(define m (layernorm (/ 5 2)))")


def test_attention_emits_projections_and_heads():
    out = compile_source("(define attn (attention 128 8))")
    assert "def init_attn(key):" in out
    assert "jax.random.split(key, 4)" in out
    assert "split_heads(x @ Wq, 8)" in out
    assert "jnp.sqrt(16)" in out
    assert "def split_heads(x, heads):" in out
    compile(out, "<generated>", "exec")


def test_attention_heads_must_divide_dim():
    with pytest.raises(CompileError, match="not divisible"):
        compile_source("(define attn (attention 128 7))")


def test_causal_attention_emits_tril_mask():
    out = compile_source("(define attn (attention 32 4 #:causal))")
    assert "jnp.tril(jnp.ones((seq, seq), dtype=bool))" in out
    assert "jnp.where(mask, scores, -jnp.inf)" in out
    compile(out, "<generated>", "exec")


def test_non_causal_attention_has_no_mask():
    out = compile_source("(define attn (attention 32 4))")
    assert "tril" not in out


def test_unknown_attention_flag_raises():
    with pytest.raises(CompileError, match="unknown attention option"):
        compile_source("(define attn (attention 32 4 #:casual))")


def test_causal_attention_ignores_future_positions(tmp_path):
    jax = pytest.importorskip("jax")
    import importlib.util

    src = "(define attn (attention 8 2 #:causal))"
    mod_path = tmp_path / "gen_causal.py"
    mod_path.write_text(compile_source(src))
    spec = importlib.util.spec_from_file_location("gen_causal", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.init_attn(jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 8))
    y = mod.attn(params, x)
    x_perturbed = x.at[-1].add(1.0)
    y_perturbed = mod.attn(params, x_perturbed)
    assert jax.numpy.allclose(y[:-1], y_perturbed[:-1], atol=1e-6)
    assert not jax.numpy.allclose(y[-1], y_perturbed[-1])


def test_attention_helpers_only_emitted_when_used():
    assert "split_heads" not in compile_source("(define m (layernorm 4))")


def test_residual_adds_input():
    out = compile_source("(define r (residual (layernorm 16)))")
    assert "def r_inner(params, x):" in out
    assert "return x + r_inner(params, x)" in out
    assert "def init_r(key):" in out


def test_residual_requires_matching_dims():
    with pytest.raises(CompileError, match="residual requires matching"):
        compile_source("(define r (residual (mlp 8 [] 4)))")


def test_let_dim_multiple_pairs():
    out = compile_source("(let-dim d 16 heads 4 ffn 4d) (define m (layernorm ffn))")
    assert "jnp.ones(64), jnp.zeros(64)" in out


def test_let_dim_grouped_bindings():
    src = """
    (let-dim
      (d 16) (heads 4) (ffn (* 4 d)))
    (define m (layernorm ffn))
    """
    out = compile_source(src)
    assert "jnp.ones(64), jnp.zeros(64)" in out


def test_let_dim_grouped_binding_wrong_shape_raises():
    with pytest.raises(CompileError, match=r"binding must be \(<name> <dim>\)"):
        compile_source("(let-dim (d 16 4)) (define m (layernorm d))")


def test_let_dim_grouped_non_symbol_name_raises():
    with pytest.raises(CompileError, match="name must be a symbol"):
        compile_source("(let-dim (4 16)) (define m (layernorm 4))")


def test_let_dim_odd_binding_count_raises():
    with pytest.raises(CompileError, match="name/dim pairs"):
        compile_source("(let-dim d 16 heads) (define m (layernorm d))")


def test_let_dim_non_symbol_name_raises():
    with pytest.raises(CompileError, match="name must be a symbol"):
        compile_source("(let-dim 4 16) (define m (layernorm 4))")


def test_defined_block_composes_in_chain():
    src = """
    (let-dim d 16)
    (define ffn (mlp d [4d] d))
    (define block (chain (layernorm d) ffn))
    """
    out = compile_source(src)
    assert "init_ffn(keys[1])" in out
    assert "x = ffn(params[1], x)" in out
    assert "block_layer1" not in out


def test_defined_block_used_twice_gets_fresh_params():
    src = "(define ffn (mlp 8 [] 8)) (define m (chain ffn ffn))"
    out = compile_source(src)
    assert "init_ffn(keys[0]), init_ffn(keys[1])" in out


def test_top_level_reference_makes_alias():
    out = compile_source("(define a (layernorm 8)) (define b a)")
    assert "def init_b(key):\n    return init_a(key)" in out
    assert "def b(params, x):\n    return a(params, x)" in out


def test_undefined_block_reference_raises():
    with pytest.raises(CompileError, match="undefined block 'ffn'"):
        compile_source("(define m (chain (layernorm 8) ffn))")


def test_reference_participates_in_dim_check():
    with pytest.raises(CompileError, match="mismatch"):
        compile_source("(define ffn (mlp 8 [] 4)) (define m (chain ffn (layernorm 5)))")


def test_residual_over_reference():
    out = compile_source("(define ffn (mlp 8 [] 8)) (define r (residual ffn))")
    assert "return x + ffn(params, x)" in out


def test_repeat_emits_body_once_and_loops():
    src = "(let-dim d 16) (define stack (repeat 3 (residual (layernorm d))))"
    out = compile_source(src)
    assert out.count("def stack_body_inner(params, x):") == 1
    assert "[init_stack_body(k) for k in jax.random.split(key, 3)]" in out
    assert "for layer_params in params:" in out


def test_repeat_count_can_be_expression():
    src = "(let-dim layers 4) (define s (repeat (* 2 layers) (layernorm 8)))"
    out = compile_source(src)
    assert "jax.random.split(key, 8)" in out


def test_repeat_over_reference():
    src = "(define blk (layernorm 8)) (define s (repeat 2 blk))"
    out = compile_source(src)
    assert "[init_blk(k) for k in jax.random.split(key, 2)]" in out


def test_repeat_requires_matching_dims():
    with pytest.raises(CompileError, match="repeat requires matching"):
        compile_source("(define s (repeat 2 (mlp 8 [] 4)))")


def test_transformer_block_compiles():
    src = """
    (let-dim d 64)
    (define block (chain
      (residual (chain (layernorm d) (attention d 4)))
      (residual (chain (layernorm d) (mlp d [4d] d)))))
    """
    out = compile_source(src)
    assert "init_dense" in out and "split_heads" in out
    compile(out, "<generated>", "exec")


def test_generated_module_runs_under_jax(tmp_path):
    jax = pytest.importorskip("jax")
    src = "(define model (chain (mlp 6 [5] 4) (layernorm 4)))"
    mod_path = tmp_path / "gen_model.py"
    mod_path.write_text(compile_source(src))
    spec = importlib.util.spec_from_file_location("gen_model", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.init_model(jax.random.PRNGKey(0))
    y = mod.model(params, jax.numpy.ones((3, 6)))
    assert y.shape == (3, 4)


def test_generated_transformer_block_runs_under_jax(tmp_path):
    jax = pytest.importorskip("jax")
    src = """
    (let-dim d 16 heads 4 layers 2)
    (define attn_block (residual (chain (layernorm d) (attention d heads))))
    (define ffn_block (residual (chain (layernorm d) (mlp d [4d] d))))
    (define block (repeat layers (chain attn_block ffn_block)))
    """
    mod_path = tmp_path / "gen_block.py"
    mod_path.write_text(compile_source(src))
    spec = importlib.util.spec_from_file_location("gen_block", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.init_block(jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 16))
    y = mod.block(params, x)
    assert y.shape == (2, 7, 16)
    assert not jax.numpy.allclose(y, x)


def run_compiler(*argv):
    return subprocess.run(
        [sys.executable, "compiler.py", *argv], capture_output=True, text=True
    )


def test_cli_writes_python_file_next_to_input(tmp_path):
    src = tmp_path / "model.jisp"
    src.write_text("(define model (mlp 4 [8] 2))")
    result = run_compiler(str(src))
    out = tmp_path / "model.py"
    assert result.returncode == 0
    assert out.exists()
    assert "def model(params, x):" in out.read_text()


def test_cli_explicit_output_path(tmp_path):
    src = tmp_path / "model.jisp"
    src.write_text("(define model (layernorm 4))")
    out = tmp_path / "custom.py"
    result = run_compiler(str(src), str(out))
    assert result.returncode == 0
    assert "def init_model(key):" in out.read_text()


def test_cli_compile_error_goes_to_stderr(tmp_path):
    src = tmp_path / "bad.jisp"
    src.write_text("(define m (conv 3))")
    result = run_compiler(str(src))
    assert result.returncode == 1
    assert "compile error" in result.stderr


def test_cli_usage_without_args():
    result = run_compiler()
    assert result.returncode == 2
    assert "usage:" in result.stderr
