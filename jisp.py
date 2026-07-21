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
