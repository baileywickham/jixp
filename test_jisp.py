import subprocess
import sys

import pytest

from jisp import (
    EvalError,
    ParseError,
    evaluate,
    exec,
    parse,
    parse_atom,
    standard_env,
    tokenize,
)


def test_tokenize_pads_parens_and_splits():
    assert tokenize("(add 1 2)") == ["(", "add", "1", "2", ")"]


def test_tokenize_handles_nesting_and_extra_whitespace():
    assert tokenize("( a\n  (b   c))") == ["(", "a", "(", "b", "c", ")", ")"]


def test_tokenize_empty_source():
    assert tokenize("   \n ") == []


def test_parse_atom_int():
    assert parse_atom("42") == 42


def test_parse_atom_negative_int():
    assert parse_atom("-7") == -7


def test_parse_atom_float():
    assert parse_atom("3.14") == 3.14


def test_parse_atom_symbol():
    assert parse_atom("add") == "add"


def test_parse_atom_symbol_that_looks_numeric_ish():
    assert parse_atom("1+") == "1+"


def test_parse_single_atom():
    assert parse("42") == [42]


def test_parse_flat_list():
    assert parse("(add 1 2.5)") == [["add", 1, 2.5]]


def test_parse_nested_list():
    assert parse("(a (b (c)) d)") == [["a", ["b", ["c"]], "d"]]


def test_parse_empty_list():
    assert parse("()") == [[]]


def test_parse_multiple_top_level_forms():
    assert parse("(a) (b 1)") == [["a"], ["b", 1]]


def test_parse_empty_source_gives_no_forms():
    assert parse("") == []


def test_parse_unexpected_close_paren():
    with pytest.raises(ParseError, match=r"unexpected '\)'"):
        parse(")")


def test_parse_unclosed_open_paren():
    with pytest.raises(ParseError, match=r"unclosed '\('"):
        parse("(a (b)")


def test_parse_brackets_give_tuples():
    assert parse("[1 2]") == [(1, 2)]


def test_parse_brackets_nest_inside_forms():
    assert parse("(mlp 8 [16 x] 4)") == [["mlp", 8, (16, "x"), 4]]


def test_parse_empty_brackets():
    assert parse("[]") == [()]


def test_parse_unclosed_bracket():
    with pytest.raises(ParseError, match=r"unclosed '\['"):
        parse("[1 2")


def test_parse_unexpected_close_bracket():
    with pytest.raises(ParseError, match=r"unexpected '\]'"):
        parse("]")


def test_parse_mismatched_delimiters():
    with pytest.raises(ParseError, match=r"'\(' closed by '\]'"):
        parse("(a]")


def test_eval_tuple_literal_evaluates_elements():
    assert run("[1 (+ 1 1) (quote a)]") == (1, 2, "a")


def run(source):
    """Parse and evaluate source in a fresh env; return the last result."""
    return exec(parse(source))


def test_eval_number_self_evaluates():
    assert evaluate(42, {}) == 42


def test_eval_symbol_lookup():
    assert evaluate("x", {"x": 3}) == 3


def test_eval_unbound_symbol_raises():
    with pytest.raises(EvalError, match="unbound symbol 'x'"):
        evaluate("x", {})


def test_eval_quote_returns_form_unevaluated():
    assert run("(quote (add 1 2))") == ["add", 1, 2]


def test_eval_if_true_branch():
    assert run("(if (< 1 2) 10 20)") == 10


def test_eval_if_false_branch():
    assert run("(if (> 1 2) 10 20)") == 20


def test_eval_define_binds_in_env():
    assert run("(define x 3) (+ x 4)") == 7


def test_eval_variadic_arithmetic():
    assert run("(- (+ 1 2 3) (* 2 2))") == 2


def test_eval_division():
    assert run("(/ 10 4)") == 2.5


def test_eval_lambda_call():
    assert run("((lambda (x y) (+ x y)) 3 4)") == 7


def test_eval_recursion_via_define():
    src = """
    (define fact (lambda (n) (if (< n 2) 1 (* n (fact (- n 1))))))
    (fact 5)
    """
    assert run(src) == 120


def test_eval_lambda_arity_mismatch_raises():
    with pytest.raises(EvalError, match="expected 2 arguments, got 1"):
        run("((lambda (x y) x) 1)")


def test_eval_calling_non_function_raises():
    with pytest.raises(EvalError, match="not callable"):
        run("(1 2)")


def test_eval_empty_list_raises():
    with pytest.raises(EvalError, match="cannot evaluate"):
        run("()")


def test_exec_returns_none_for_no_forms():
    assert exec([]) is None


def test_exec_shares_env_across_forms():
    env = standard_env()
    exec(parse("(define x 1)"), env)
    assert exec(parse("(+ x 1)"), env) == 2


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, "jisp.py", *argv], capture_output=True, text=True
    )


def test_cli_evaluates_file(tmp_path):
    src = tmp_path / "prog.lisp"
    src.write_text("(define x 3)\n(+ x 4)\n(quote (a b))\n")
    result = run_cli(str(src))
    assert result.returncode == 0
    assert result.stdout == "7\n['a', 'b']\n"


def test_cli_eval_error_goes_to_stderr(tmp_path):
    src = tmp_path / "bad.lisp"
    src.write_text("(boom 1)")
    result = run_cli(str(src))
    assert result.returncode == 1
    assert "unbound symbol 'boom'" in result.stderr


def test_cli_parse_error_goes_to_stderr(tmp_path):
    src = tmp_path / "bad.lisp"
    src.write_text("(a (b)")
    result = run_cli(str(src))
    assert result.returncode == 1
    assert "unclosed '('" in result.stderr


def test_cli_usage_without_args():
    result = run_cli()
    assert result.returncode == 2
    assert "usage:" in result.stderr
