import dataclasses

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx


def process_signature(app, what, name, obj, options, signature, return_annotation):
    if what != "class":
        return None

    if not dataclasses.is_dataclass(obj):
        return None

    if signature is None or "<factory>" not in signature:
        return None

    factory_fields = [
        f
        for f in dataclasses.fields(obj)
        if f.default_factory is not dataclasses.MISSING and f.init
    ]

    for f in factory_fields:
        factory_name = getattr(f.default_factory, "__name__", repr(f.default_factory))
        signature = signature.replace("<factory>", f"{factory_name}()", 1)

    return (signature, return_annotation)


def _inline_attr_types(doctree):
    for desc_node in doctree.traverse(addnodes.desc):
        if desc_node.get("objtype") != "attribute":
            continue

        sigs = list(desc_node.traverse(addnodes.desc_signature))
        contents = list(desc_node.traverse(addnodes.desc_content))
        if not sigs or not contents:
            continue

        sig = sigs[0]
        content = contents[0]

        type_nodes = None
        type_field = None
        type_field_list = None

        for fl in content.traverse(nodes.field_list):
            for field in list(fl.traverse(nodes.field)):
                fn = field.traverse(nodes.field_name)
                if fn and fn[0].astext().strip() == "Type":
                    fb = field.traverse(nodes.field_body)
                    if fb:
                        paras = fb[0].traverse(nodes.paragraph)
                        if paras:
                            type_nodes = list(paras[0].children)
                    type_field = field
                    type_field_list = fl
                    break
            if type_nodes is not None:
                break

        if type_nodes is None:
            continue

        insert_idx = len(sig.children)
        for i, child in enumerate(sig.children):
            if isinstance(child, nodes.reference) and "headerlink" in child.get(
                "classes", []
            ):
                insert_idx = i
                break

        new_nodes = [
            addnodes.desc_sig_space("", " "),
            addnodes.desc_sig_punctuation("", ":"),
            addnodes.desc_sig_space("", " "),
        ]
        for tn in type_nodes:
            new_nodes.append(tn.deepcopy())

        for j, nn in enumerate(new_nodes):
            sig.insert(insert_idx + j, nn)

        type_field.parent.remove(type_field)
        if not type_field_list.children:
            type_field_list.parent.remove(type_field_list)


def _remove_attrs_from_toc(app, doctree, docname):
    attr_ids = set()
    for desc in doctree.traverse(addnodes.desc):
        if desc.get("objtype") == "attribute":
            for sig in desc.traverse(addnodes.desc_signature):
                attr_ids.update(sig.get("ids", []))

    if not attr_ids:
        return

    toc = app.env.tocs.get(docname)
    if toc is None:
        return

    to_remove = []
    for ref in toc.traverse(nodes.reference):
        refid = ref.get("anchorname", "").lstrip("#") or ref.get("refid", "")
        if refid in attr_ids:
            list_item = ref.parent.parent
            if isinstance(list_item, nodes.list_item):
                to_remove.append(list_item)

    for item in to_remove:
        if item.parent is not None:
            item.parent.remove(item)


def process_doctree(app, doctree, docname):
    _inline_attr_types(doctree)
    _remove_attrs_from_toc(app, doctree, docname)


def setup(app: Sphinx):
    app.connect("autodoc-process-signature", process_signature)
    app.connect("doctree-resolved", process_doctree)
