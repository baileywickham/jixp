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
