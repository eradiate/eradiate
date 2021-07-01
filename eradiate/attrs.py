"""attrs-based utility classes and functions"""

import enum
from textwrap import dedent, indent

import attr


class _Auto:
    """
    Sentinel class to indicate when a dynamic field value is expected to be
    set automatically. ``_Auto`` is a singleton. There is only ever one of it.

    .. note:: ``bool(_Auto)`` evaluates to ``False``.
    """

    _singleton = None

    def __new__(cls):
        if _Auto._singleton is None:
            _Auto._singleton = super(_Auto, cls).__new__(cls)
        return _Auto._singleton

    def __repr__(self):
        return "AUTO"

    def __bool__(self):
        return False

    def __len__(self):
        return 0


AUTO = _Auto()
"""
Sentinel to indicate when a dynamic field value is expected to be set automatically.
"""


class MetadataKey(enum.Enum):
    """Attribute metadata keys.

    These Enum values should be used as metadata attribute keys.
    """

    DOC = enum.auto()  #: Documentation for this field (str)
    TYPE = enum.auto()  #: Documented type for this field (str)
    DEFAULT = enum.auto()  #: Documented default value for this field (str)


# ------------------------------------------------------------------------------
#                           Attribute docs extension
# ------------------------------------------------------------------------------


@attr.s
class _FieldDoc:
    """Internal convenience class to store field documentation information."""

    doc = attr.ib(default=None)
    type = attr.ib(default=None)
    default = attr.ib(default=None)


def _eradiate_formatter(cls_doc, field_docs):
    """Appends a section on attributes to a class docstring.
    This docstring formatter is appropriate for Eradiate's current docstring
    format.

    Parameter ``cls_doc`` (str):
        Class docstring to extend.

    Parameter ``field_docs`` (dict[str, _FieldDoc]):
        Attributes documentation content.

    Returns → str:
        Updated class docstring.
    """
    # Do nothing if field is not documented
    if not field_docs:
        return cls_doc

    docstrings = []

    # Create docstring entry for each documented field
    for field_name, field_doc in field_docs.items():
        type_doc = f": {field_doc.type}" if field_doc.type is not None else ""
        default_doc = f" = {field_doc.default}" if field_doc.default is not None else ""

        docstrings.append(
            f"``{field_name.lstrip('_')}``{type_doc}{default_doc}\n"
            f"{indent(field_doc.doc, '    ')}\n"
        )

    # Assemble entries
    if docstrings:
        if cls_doc is None:
            cls_doc = ""

        return "\n".join(
            (
                dedent(cls_doc.lstrip("\n")).rstrip(),
                "",
                ".. rubric:: Constructor arguments / instance attributes",
                "",
                "\n".join(docstrings),
                "",
            )
        )
    else:
        return cls_doc


def parse_docs(cls):
    """Extract attribute documentation and update class docstring with it.

    .. admonition:: Notes

       * Meant to be used as a class decorator.
       * Must be applied **after** ``@attr.s``.
       * Fields must be documented using :func:`documented`.

    This decorator will examine each ``attrs`` attribute and check its metadata
    for documentation content. It will then update the class's docstring
    based on this content.

    .. seealso:: :func:`documented`

    Parameter ``cls`` (class):
        Class whose attributes should be processed.

    Returns → class:
        Updated class.
    """
    formatter = _eradiate_formatter

    docs = {}
    for field in cls.__attrs_attrs__:
        if MetadataKey.DOC in field.metadata:
            # Collect field docstring
            docs[field.name] = _FieldDoc(doc=field.metadata[MetadataKey.DOC])

            # Collect field type
            if MetadataKey.TYPE in field.metadata:
                docs[field.name].type = field.metadata[MetadataKey.TYPE]
            else:
                docs[field.name].type = str(field.type)

            # Collect default value
            if MetadataKey.DEFAULT in field.metadata:
                docs[field.name].default = field.metadata[MetadataKey.DEFAULT]

    # Update docstring
    cls.__doc__ = formatter(cls.__doc__, docs)

    return cls


def documented(attrib, doc=None, type=None, default=None):
    """Declare an attrs field as documented.

    .. seealso:: :func:`parse_docs`

    Parameter ``doc`` (str or None):
        Docstring for the considered field. If set to ``None``, this function
        does nothing.

    Parameter ``type`` (str or None):
        Documented type for the considered field.

    Parameter ``default`` (str or None):
        Documented default value for the considered field.

    Returns → ``attrs`` attribute:
        ``attrib``, with metadata updated with documentation contents.
    """
    if doc is not None:
        attrib.metadata[MetadataKey.DOC] = doc

    if type is not None:
        attrib.metadata[MetadataKey.TYPE] = type

    if default is not None:
        attrib.metadata[MetadataKey.DEFAULT] = default

    return attrib


def get_doc(cls, attrib, field):
    """Fetch attribute documentation field. Requires fields metadata to be
    processed with :func:`documented`.

    Parameter ``cls`` (class):
        Class from which to get the attribute.

    Parameter ``attrib`` (str):
        Attribute from which to get the doc field.

    Parameter ``field`` ("doc" or "type" or "default"):
        Documentation field to query.

    Returns:
        Queried documentation content.

    Raises → ValueError:
        If the requested ``field`` is missing from the target attribute's
        metadata.

    Raises → ValueError:
        If the requested ``field`` is unsupported.
    """
    try:
        if field == "doc":
            return attr.fields_dict(cls)[attrib].metadata[MetadataKey.DOC]

        if field == "type":
            return attr.fields_dict(cls)[attrib].metadata[MetadataKey.TYPE]

        if field == "default":
            return attr.fields_dict(cls)[attrib].metadata[MetadataKey.DEFAULT]
    except KeyError:
        raise ValueError(
            f"{cls.__name__}.{attrib} has no documented field " f"'{field}'"
        )

    raise ValueError(f"unsupported attribute doc field {field}")
