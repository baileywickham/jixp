# jisp Minimal Lisp Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-file Python parser for basic Lisp s-expressions, usable as a library (`parse()`) and as a file CLI (`python jisp.py file.lisp`).

**Architecture:** Split-based tokenizer (pad parens, split on whitespace) feeding a recursive reader that builds nested Python lists. Atoms become `int`/`float`/`str`. A `ParseError` exception covers the two syntax errors possible in this grammar: unexpected `)` and unclosed `(`.

**Tech Stack:** Python 3 stdlib only. pytest for tests.

## Global Constraints

- Everything lives in `jisp.py`; tests in `test_jisp.py` (repo root, no packages).
- No third-party runtime dependencies; pytest is the only dev dependency.
- Data representation: `list` for s-expressions, `int`/`float` for numbers, `str` for symbols.
- Parsing only — no evaluation, strings, quotes, or comments (spec: out of scope).
- Spec: `docs/superpowers/specs/2026-07-21-lisp-parser-design.md`.

---

### Task 1: Tokenizer and atom parsing

**Files:**
- Create: `jisp.py`
- Test: `test_jisp.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `tokenize(source: str) -> list[str]`, `parse_atom(token: str) -> int | float | str`, and the exception class `ParseError(Exception)` (defined now so Task 2 can raise it).

- [ ] **Step 1: Write the failing tests**

Create `test_jisp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_jisp.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'jisp'`

- [ ] **Step 3: Write minimal implementation**

Create `jisp.py`:

```python
"""jisp: a minimal Lisp s-expression parser."""


class ParseError(Exception):
    """Raised when source text is not a valid s-expression."""


def tokenize(source):
    """Split Lisp source into paren and atom tokens."""
    return source.replace("(", " ( ").replace(")", " ) ").split()


def parse_atom(token):
    """Convert a token to int or float if possible, else keep it as a symbol."""
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_jisp.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add jisp.py test_jisp.py
git commit -m "feat: add tokenizer and atom parsing"
```

---

### Task 2: Recursive reader (`parse`)

**Files:**
- Modify: `jisp.py`
- Test: `test_jisp.py`

**Interfaces:**
- Consumes: `tokenize`, `parse_atom`, `ParseError` from Task 1 (signatures as defined there).
- Produces: `parse(source: str) -> list` returning a list of top-level forms; internal helper `_read_form(tokens: list[str], pos: int) -> tuple[object, int]`. Raises `ParseError` with message `"unexpected ')'"` or `"unclosed '('"`.

- [ ] **Step 1: Write the failing tests**

Append to `test_jisp.py`:

```python
import pytest

from jisp import ParseError, parse


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_jisp.py -v`
Expected: collection error — `ImportError: cannot import name 'parse' from 'jisp'`

- [ ] **Step 3: Write minimal implementation**

Append to `jisp.py`:

```python
def parse(source):
    """Parse Lisp source into a list of top-level forms."""
    tokens = tokenize(source)
    forms = []
    pos = 0
    while pos < len(tokens):
        form, pos = _read_form(tokens, pos)
        forms.append(form)
    return forms


def _read_form(tokens, pos):
    """Read one form starting at tokens[pos]; return (form, next_pos)."""
    token = tokens[pos]
    if token == "(":
        items = []
        pos += 1
        while True:
            if pos >= len(tokens):
                raise ParseError("unclosed '('")
            if tokens[pos] == ")":
                return items, pos + 1
            item, pos = _read_form(tokens, pos)
            items.append(item)
    if token == ")":
        raise ParseError("unexpected ')'")
    return parse_atom(token), pos + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_jisp.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add jisp.py test_jisp.py
git commit -m "feat: add recursive s-expression reader"
```

---

### Task 3: File CLI

**Files:**
- Modify: `jisp.py`
- Test: `test_jisp.py`

**Interfaces:**
- Consumes: `parse` and `ParseError` from Task 2.
- Produces: `main() -> None` and an `if __name__ == "__main__"` guard so `python jisp.py file.lisp` pretty-prints each parsed form. Exit codes: 0 on success, 1 on `ParseError` (message on stderr), 2 on wrong argument count (usage on stderr).

- [ ] **Step 1: Write the failing tests**

Append to `test_jisp.py`:

```python
import subprocess
import sys


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test_jisp.py -v`
Expected: the 3 new CLI tests FAIL (`returncode` 0 vs expected 2, empty stdout — the script has no `main`), 16 prior tests still pass

- [ ] **Step 3: Write minimal implementation**

Append to `jisp.py` (add `import sys` and `from pprint import pprint` at the top of the file, below the docstring):

```python
import sys
from pprint import pprint
```

```python
def main():
    if len(sys.argv) != 2:
        print("usage: python jisp.py file.lisp", file=sys.stderr)
        sys.exit(2)
    with open(sys.argv[1]) as f:
        source = f.read()
    try:
        forms = parse(source)
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        sys.exit(1)
    for form in forms:
        pprint(form)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test_jisp.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add jisp.py test_jisp.py
git commit -m "feat: add file CLI"
```
