from docutils import nodes


FALLBACK_ROLES = ("class", "data", "attribute", "obj")

INTERNAL_MODULE_ALIASES = {
    "pathlib._local.Path": "pathlib.Path",
    "pathlib._local.PurePath": "pathlib.PurePath",
}


def _resolve_type_aliases(app, env, node, contnode):
    """Retry failed :class: lookups as :data: for type aliases like NDArray.

    Sphinx resolves type annotations as :class:, but some types (e.g.
    numpy.typing.NDArray) are registered as :data: in intersphinx inventories.
    This handler catches the mismatch and retries with the correct role.

    Also normalizes internal CPython module paths (e.g. pathlib._local.Path)
    to their public API equivalents (pathlib.Path).
    """
    if node.get("refdomain") != "py" or node.get("reftype") != "class":
        return None

    target = node["reftarget"]
    target = INTERNAL_MODULE_ALIASES.get(target, target)
    named_inv = getattr(env, "intersphinx_named_inventory", {})

    for _proj_name, proj_inv in named_inv.items():
        for role in FALLBACK_ROLES:
            key = f"py:{role}"
            if key in proj_inv and target in proj_inv[key]:
                _proj, _version, uri, _dispname = proj_inv[key][target]
                short_name = target.rsplit(".", 1)[-1]
                newnode = nodes.reference(
                    short_name, short_name, internal=False, refuri=uri
                )
                return newnode

    return None


def setup(app):
    app.connect("missing-reference", _resolve_type_aliases)
