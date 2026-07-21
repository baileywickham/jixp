# jisp: minimal Lisp parser — design

Date: 2026-07-21
Status: approved

## Goal

A single-file Python parser for basic Lisp s-expressions, usable as a library
and as a small file-parsing CLI. Parsing only — no evaluation.

## Scope

- Atoms: integers, floats, symbols (everything else is a symbol, kept as `str`).
- Nested lists via `(` and `)`.
- Multiple top-level forms per source string.
- Out of scope (for now): strings, quote sugar, comments, booleans, evaluation.

## Approach

Split-based tokenizer + recursive reader (Norvig-style). Chosen over a
regex/stack parser (better diagnostics but ~2x code) and a character-stream
reader (only pays off once strings/comments are added).

## Components (all in `jisp.py`)

- `ParseError(Exception)` — raised on syntax errors.
- `tokenize(source: str) -> list[str]` — pads parens with spaces, splits on
  whitespace.
- `parse_atom(token: str) -> int | float | str` — tries `int`, then `float`,
  falls back to symbol (`str`).
- `parse(source: str) -> list` — returns a list of top-level forms, so
  `"(a) (b)"` parses to two forms. Raises `ParseError` on an unexpected `)`
  or an unclosed `(`.
- CLI: `python jisp.py file.lisp` reads the file and pretty-prints each parsed
  form. Parse errors go to stderr with exit code 1.

## Data representation

Plain Python: `list` for s-expressions, `int`/`float` for numbers, `str` for
symbols.

## Testing

`test_jisp.py` (pytest): atoms of each type, nesting, multiple top-level
forms, unexpected `)`, unclosed `(`.
