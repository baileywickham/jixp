"""jisp: a minimal Lisp interpreter — s-expression parser and evaluator."""

import operator
import sys
from functools import reduce
from pprint import pprint


class ParseError(Exception):
    """Raised when source text is not a valid s-expression."""


class EvalError(Exception):
    """Raised when a parsed form cannot be evaluated."""


def tokenize(source):
    """Split Lisp source into paren, bracket, and atom tokens."""
    for delim in "()[]":
        source = source.replace(delim, f" {delim} ")
    return source.split()


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
    """Read one form starting at tokens[pos]; return (form, next_pos).

    Parens read as forms (Python lists); brackets read as data tuples.
    """
    token = tokens[pos]
    if token in ("(", "["):
        closer = ")" if token == "(" else "]"
        items = []
        pos += 1
        while True:
            if pos >= len(tokens):
                raise ParseError(f"unclosed '{token}'")
            if tokens[pos] in (")", "]"):
                if tokens[pos] != closer:
                    raise ParseError(f"'{token}' closed by '{tokens[pos]}'")
                return (items if closer == ")" else tuple(items)), pos + 1
            item, pos = _read_form(tokens, pos)
            items.append(item)
    if token in (")", "]"):
        raise ParseError(f"unexpected '{token}'")
    return parse_atom(token), pos + 1


def standard_env():
    """Build a fresh global environment with the builtin procedures."""
    return {
        "+": lambda *args: sum(args),
        "-": lambda first, *rest: (
            -first if not rest else reduce(operator.sub, rest, first)
        ),
        "*": lambda *args: reduce(operator.mul, args, 1),
        "/": lambda first, *rest: (
            1 / first if not rest else reduce(operator.truediv, rest, first)
        ),
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "=": operator.eq,
    }


def evaluate(form, env):
    """Evaluate one parsed form in env."""
    match form:
        case int() | float():
            return form
        case str() as name:
            try:
                return env[name]
            except KeyError:
                raise EvalError(f"unbound symbol '{name}'") from None
        case tuple():
            # Bracket literals are data: evaluate elements, keep the tuple.
            # Must precede the list patterns below, which also match tuples.
            return tuple(evaluate(item, env) for item in form)
        case ["quote", quoted]:
            return quoted
        case ["if", test, then, alt]:
            return evaluate(then if evaluate(test, env) else alt, env)
        case ["define", str() as name, value]:
            env[name] = evaluate(value, env)
        case ["lambda", [*params], body]:

            def procedure(*args):
                if len(args) != len(params):
                    raise EvalError(
                        f"expected {len(params)} arguments, got {len(args)}"
                    )
                return evaluate(body, {**env, **dict(zip(params, args))})

            return procedure
        case [op, *args]:
            fn = evaluate(op, env)
            if not callable(fn):
                raise EvalError(f"not callable: {fn!r}")
            return fn(*[evaluate(arg, env) for arg in args])
        case _:
            raise EvalError(f"cannot evaluate {form!r}")


def exec(forms, env=None):
    """Evaluate forms in order (sharing one env); return the last result."""
    if env is None:
        env = standard_env()
    result = None
    for form in forms:
        result = evaluate(form, env)
    return result


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
    env = standard_env()
    try:
        for form in forms:
            result = evaluate(form, env)
            if result is not None:
                pprint(result)
    except EvalError as exc:
        print(f"eval error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
