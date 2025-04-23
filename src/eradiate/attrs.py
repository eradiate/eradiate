"""*attrs*-based utility classes and functions."""

from __future__ import annotations

import enum
import re
import typing as t
from textwrap import dedent, indent

import attrs

from .util import numpydoc


class _Auto:
    """
    Sentinel class to indicate when a dynamic field value is expected to be
    set automatically. ``_Auto`` is a singleton. There is only ever one of it.

    Notes
    -----
    ``bool(_Auto)`` evaluates to ``False``.
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


#: Typing alias for :data:`.AUTO`.
AutoType = _Auto

#: Sentinel to indicate when a dynamic field value is expected to be set
#: automatically.
AUTO = _Auto()


class MetadataKey(enum.Enum):
    """
    Attribute metadata keys.

    These Enum values should be used as metadata attribute keys.
    """

    DOC = "doc"  #: Documentation for this field (str)
    TYPE = "type"  #: Documented type for this field (str)
    INIT_TYPE = "init_type"  #: Documented constructor parameter for this field (str)
    DEFAULT = "default"  #: Documented default value for this field (str)


class DocFlags(enum.Flag):
    """
    Extra flags used to pass information about docs.
    """

    NOINIT = 0


# ------------------------------------------------------------------------------
#                           Attribute docs extension
# ------------------------------------------------------------------------------


@attrs.define
class _FieldDoc:
    """Internal convenience class to store field documentation information."""

    doc = attrs.field(default=None)
    type = attrs.field(default=None)
    init_type = attrs.field(default=None)
    default = attrs.field(default=None)


def _eradiate_formatter(cls_doc: str, field_docs: dict[str, _FieldDoc]) -> str:
    """
    Appends a section on attributes to a class docstring.
    This docstring formatter is appropriate for Eradiate's current docstring
    format.

    Parameters
    ----------
    cls_doc : str
        Class docstring to extend.

    field_docs : dict[str, _FieldDoc]
        Attributes documentation content.

    Returns
    -------
    str
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


def _numpy_formatter(cls_doc: str, field_docs: dict[str, _FieldDoc]) -> str:
    """
    Append a section on attributes to a class docstring.
    This docstring formatter is appropriate for the Numpy docstring format.

    Parameters
    ----------
    cls_doc : str
        Class docstring to extend.

    field_docs : dict[str, _FieldDoc]
        Attributes documentation content.

    Returns
    -------
    str
        Updated class docstring.
    """
    # Do nothing if no field is documented
    if not field_docs:
        return cls_doc

    param_docstrings = []
    attr_docstrings = []

    # Create docstring entry for each documented field
    for field_name, field_doc in field_docs.items():
        field_type = field_doc.type
        field_init_type = field_doc.init_type

        # Generate constructor parameter docstring entry
        if field_init_type is not DocFlags.NOINIT:
            init_type_doc = (
                f" : {field_init_type}" if field_init_type is not None else ""
            )
            default_doc = (
                f", default: {field_doc.default}"
                if field_doc.default is not None
                else ""
            )
            param_docstrings.append(
                f"{field_name.lstrip('_')}{init_type_doc}{default_doc}\n"
                f"{indent(field_doc.doc, '    ')}\n"
            )

            # Generate attribute docstring entry
            type_doc = field_type if field_type is not None else ""
            if not field_name.startswith("_"):
                field_doc_brief = re.split(r"\. |\.\n", field_doc.doc)[0].strip()
                if not field_doc_brief.endswith("."):
                    field_doc_brief += "."
                attr_docstrings.append(
                    f"{field_name} : {type_doc}\n{indent(field_doc_brief, '    ')}\n"
                )

    # Assemble entries
    if param_docstrings or attr_docstrings:
        if cls_doc is None:
            cls_doc = ""

        cls_doc = dedent(cls_doc.lstrip("\n")).rstrip()

        # Parse docstrings
        sections = numpydoc.parse_doc(cls_doc)

        # Append generated docstrings to the relevant section
        sections["Parameters"] = "\n".join(param_docstrings) + sections.get(
            "Parameters", ""
        )
        sections["Fields"] = "\n".join(attr_docstrings) + sections.get("Fields", "")

        # Generate section full text
        doc = numpydoc.format_doc(sections)
        return doc

    else:
        return cls_doc


def parse_docs(cls: type) -> type:
    """
    Extract attribute documentation and update class docstring with it.

    This decorator will examine each *attrs* attribute and check its metadata
    for documentation content. It will then update the class's docstring
    based on this content.

    Parameters
    ----------
    cls : type
        Class whose attributes should be processed.

    Returns
    -------
    type
        Updated class.

    See Also
    --------
    :func:`documented` : Field documentation definition function.

    Notes
    -----
    * Meant to be used as a class decorator.
    * Must be applied **after** :func:`@attr.s <attr.s>` (or any other
      attrs decorator).
    * Fields must be documented using :func:`documented`.
    """
    formatter = _numpy_formatter

    docs = {}
    for field in cls.__attrs_attrs__:
        # Collect documentation entries
        if MetadataKey.DOC in field.metadata:
            # Collect field docstring
            docs[field.name] = _FieldDoc(doc=field.metadata[MetadataKey.DOC])

            # Collect field type
            if MetadataKey.TYPE in field.metadata:
                docs[field.name].type = field.metadata[MetadataKey.TYPE]
            else:
                docs[field.name].type = str(field.type)

            # Collect field init type
            if MetadataKey.INIT_TYPE in field.metadata:
                docs[field.name].init_type = field.metadata[MetadataKey.INIT_TYPE]
            elif field.init is False:
                docs[field.name].init_type = DocFlags.NOINIT
            else:
                docs[field.name].init_type = docs[field.name].type

            # Collect default value
            if MetadataKey.DEFAULT in field.metadata:
                docs[field.name].default = field.metadata[MetadataKey.DEFAULT]

    # Update docstring
    cls.__doc__ = formatter(cls.__doc__, docs)

    return cls


def documented(
    attrib: attrs.Attribute,
    doc: str | None = None,
    type: str | None = None,
    init_type: str | None = None,
    default: str | None = None,
) -> attrs.Attribute:
    """
    Declare an attrs field as documented.

    Parameters
    ----------
    attrib : attrs.Attribute
        *attrs* attribute definition to which documentation is to be attached.

    doc : str, optional
        Docstring for the considered field. If set to ``None``, this function
        does nothing.

    type : str, optional
        Documented type for the considered field.

    init_type : str, optional
        Documented constructor parameter for the considered field.

    default : str, optional
        Documented default value for the considered field.

    Returns
    -------
    attrs.Attribute
        ``attrib``, with metadata updated with documentation contents.

    See Also
    --------
    :func:`attrs.field`, :func:`pinttr.field`, :func:`parse_docs`
    """
    if doc is not None:
        attrib.metadata[MetadataKey.DOC] = doc

    if type is not None:
        attrib.metadata[MetadataKey.TYPE] = type

    if init_type is not None:
        attrib.metadata[MetadataKey.INIT_TYPE] = init_type

    if default is not None:
        attrib.metadata[MetadataKey.DEFAULT] = default

    return attrib


def get_doc(
    cls: type, attrib: str, field: t.Literal["doc", "type", "init_type", "default"]
) -> str:
    """
    Fetch attribute documentation field. Requires field metadata to be processed
    with :func:`documented`.

    Parameters
    ----------
    cls : type
        Class from which to get the attribute.

    attrib : str
        Attribute from which to get the doc field.

    field : {"doc", "type", "init_type", "default"}
        Documentation field to query.

    Returns
    -------
    str
        Queried documentation content.

    Raises
    ------
    ValueError
        If the requested ``field`` is missing from the target attribute's
        metadata.

    ValueError
        If the requested ``field`` is unsupported.
    """
    try:
        key = MetadataKey[field.upper()]
    except ValueError:
        raise ValueError(f"unsupported attribute doc field {field}")

    try:
        return attrs.fields_dict(cls)[attrib].metadata[key]
    except KeyError:
        raise ValueError(f"{cls.__name__}.{attrib} has no documented field '{field}'")

    # This is very unlikely to happen, but just in case, we raise
    raise RuntimeError


def define(maybe_cls=None, **kwargs):
    """
    A wrapper around :func:`attrs.define` that automatically applies docstring
    processing with :func:`.parse_docs`. All arguments are forwarded to
    :func:`attrs.define`.
    """

    def wrap(cls):
        return parse_docs(attrs.define(maybe_cls=cls, **kwargs))

    return wrap if maybe_cls is None else wrap(maybe_cls)


def frozen(maybe_cls=None, **kwargs):
    """
    A wrapper around :func:`attrs.frozen` that automatically applies docstring
    processing with :func:`.parse_docs`. All arguments are forwarded to
    :func:`attrs.frozen`.
    """

    def wrap(cls):
        return parse_docs(attrs.frozen(maybe_cls=cls, **kwargs))

    return wrap if maybe_cls is None else wrap(maybe_cls)
