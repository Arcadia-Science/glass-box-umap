from sphinx.util.inspect import TypeAliasForwardRef


def _stringify_arg(arg: object) -> str:
    if isinstance(arg, tuple):
        return ", ".join(_stringify_arg(a) for a in arg)
    if isinstance(arg, TypeAliasForwardRef):
        return arg.name
    if isinstance(arg, type):
        if arg.__module__ == "builtins":
            return arg.__qualname__
        return f"{arg.__module__}.{arg.__qualname__}"
    return repr(arg)


def _getitem(self: TypeAliasForwardRef, item: object) -> TypeAliasForwardRef:
    return TypeAliasForwardRef(f"{self.name}[{_stringify_arg(item)}]")


def _or(self: TypeAliasForwardRef, other: object) -> TypeAliasForwardRef:
    return TypeAliasForwardRef(f"{self.name} | {_stringify_arg(other)}")


def _ror(self: TypeAliasForwardRef, other: object) -> TypeAliasForwardRef:
    return TypeAliasForwardRef(f"{_stringify_arg(other)} | {self.name}")


def setup(app):
    TypeAliasForwardRef.__getitem__ = _getitem
    TypeAliasForwardRef.__or__ = _or
    TypeAliasForwardRef.__ror__ = _ror
    return {"parallel_read_safe": True}
