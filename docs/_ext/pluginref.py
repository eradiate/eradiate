from docutils import nodes
from sphinx import addnodes
from sphinx.util.nodes import split_explicit_title


def plugin_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Role for cross-referencing Eradiate Mitsuba plugins by their ID.

    Usage::

        :plugin:`bsdf-bilambertian`
        :plugin:`Bilambertian BSDF <bsdf-bilambertian>`

    The target ``bsdf-bilambertian`` is expanded to the intersphinx label
    ``plugin-bsdf-bilambertian`` in the ``eradiatemitsuba`` inventory.
    The default display text is the plugin name (last dash-separated segment).

    """
    options = options or {}
    content = content or {}

    env = inliner.document.settings.env
    has_explicit, display, target = split_explicit_title(text)
    label = f"plugin-{target}"
    if not has_explicit:
        display = target.rsplit("-", 1)[-1]

    refnode = addnodes.pending_xref(
        rawtext,
        refdoc=env.docname,
        refdomain="std",
        reftype="ref",
        reftarget=label,
        refexplicit=has_explicit,
    )
    refnode += nodes.inline(rawtext, display, classes=["xref", "plugin"])
    return [refnode], []


def setup(app):
    app.add_role("plugin", plugin_role)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
