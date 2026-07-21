import importlib.util
import subprocess
import sys

import pytest

from compiler import CompileError, compile_source

MLP_SRC = "(define model (mlp 784 (256 256) 10))"


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
    out = compile_source("(define lin (mlp 8 () 4))")
    assert "jax.random.split(key, 1)" in out
    assert "init_dense(keys[0], 8, 4)" in out


def test_layernorm_emits_scale_and_shift():
    out = compile_source("(define norm (layernorm 128))")
    assert "def init_norm(key):" in out
    assert "jnp.ones(128), jnp.zeros(128)" in out
    assert "init_dense" not in out


def test_chain_emits_sublayers_and_threads_x():
    out = compile_source("(define model (chain (mlp 784 (256) 128) (layernorm 128)))")
    assert "def model_layer0(params, x):" in out
    assert "def model_layer1(params, x):" in out
    assert "x = model_layer0(params[0], x)" in out
    assert "x = model_layer1(params[1], x)" in out
    compile(out, "<generated>", "exec")


def test_chain_dimension_mismatch_raises():
    with pytest.raises(CompileError, match="mismatch"):
        compile_source("(define m (chain (mlp 8 () 4) (layernorm 5)))")


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


def test_generated_module_runs_under_jax(tmp_path):
    jax = pytest.importorskip("jax")
    src = "(define model (chain (mlp 6 (5) 4) (layernorm 4)))"
    mod_path = tmp_path / "gen_model.py"
    mod_path.write_text(compile_source(src))
    spec = importlib.util.spec_from_file_location("gen_model", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.init_model(jax.random.PRNGKey(0))
    y = mod.model(params, jax.numpy.ones((3, 6)))
    assert y.shape == (3, 4)


def run_compiler(*argv):
    return subprocess.run(
        [sys.executable, "compiler.py", *argv], capture_output=True, text=True
    )


def test_cli_writes_python_file_next_to_input(tmp_path):
    src = tmp_path / "model.lisp"
    src.write_text("(define model (mlp 4 (8) 2))")
    result = run_compiler(str(src))
    out = tmp_path / "model.py"
    assert result.returncode == 0
    assert out.exists()
    assert "def model(params, x):" in out.read_text()


def test_cli_explicit_output_path(tmp_path):
    src = tmp_path / "model.lisp"
    src.write_text("(define model (layernorm 4))")
    out = tmp_path / "custom.py"
    result = run_compiler(str(src), str(out))
    assert result.returncode == 0
    assert "def init_model(key):" in out.read_text()


def test_cli_compile_error_goes_to_stderr(tmp_path):
    src = tmp_path / "bad.lisp"
    src.write_text("(define m (conv 3))")
    result = run_compiler(str(src))
    assert result.returncode == 1
    assert "compile error" in result.stderr


def test_cli_usage_without_args():
    result = run_compiler()
    assert result.returncode == 2
    assert "usage:" in result.stderr
