import subprocess
import sys

import pytest

from jisp import ParseError, parse, parse_atom, tokenize


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


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, "jisp.py", *argv], capture_output=True, text=True
    )


def test_cli_parses_file(tmp_path):
    src = tmp_path / "prog.lisp"
    src.write_text("(add 1 2)\n(sub 3 4)\n")
    result = run_cli(str(src))
    assert result.returncode == 0
    assert result.stdout == "['add', 1, 2]\n['sub', 3, 4]\n"


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
