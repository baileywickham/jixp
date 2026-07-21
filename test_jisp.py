from jisp import parse_atom, tokenize


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
