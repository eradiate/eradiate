"""
Numpydoc docstring tools.
"""

from __future__ import annotations

import re
from textwrap import dedent

#: List of support docstring sections. The "Fields" section is unique to
#: Eradiate.
NUMPYDOC_SECTION_TITLES = [
    "Parameters",
    "Fields",  # Special section used for classes only
    "Attributes",
    "Returns",
    "Yields",
    "Receives",
    "Other Parameters",
    "Raises",
    "Warns",
    "Warnings",
    "See Also",
    "Notes",
    "References",
    "Examples",
]


def parse_doc(doc: str) -> dict[str, str]:
    """
    Parse a docstring formatted with the Numpydoc style.

    Parameters
    ----------
    doc : str
        Docstring to parse.

    Returns
    -------
    sections : dict
        A dictionary mapping section names to their content. Special
        ``_short_summary`` and ``_extended_summary`` keys are used for
        summary contents.
    """
    doc = dedent(doc.lstrip("\n")).rstrip()

    # We process only sections mentioned in the Numpy doc style
    # Detect existing sections
    section_pattern = {}

    for section_title in NUMPYDOC_SECTION_TITLES:
        full_section_title = f"{section_title}\n{'-' * len(section_title)}\n"
        if re.findall(full_section_title, doc):
            section_pattern[section_title] = full_section_title

    # Collect section content
    section_contents = (
        re.split("|".join(section_pattern.values()), doc) if section_pattern else [doc]
    )

    sections = {}
    if len(section_contents) == len(section_pattern) + 1:
        summary = section_contents.pop(0).strip().split("\n\n")
        sections["_short_summary"] = summary.pop(0)
        if summary:
            sections["_extended_summary"] = "\n\n".join(summary)

    for title, content in zip(section_pattern.keys(), section_contents):
        sections[title] = content.strip()

    return sections


def format_doc(sections: dict[str, str]) -> str:
    """
    Assemble docstring sections and format them according to the Numpydoc style.

    Parameters
    ----------
    sections : dict
        A dictionary mapping section names to contents. Section ordering
        follows the Nompydoc style guide.

    Returns
    -------
    docstring : str
        Formatted docstring.
    """

    # Generate section full text
    section_fulltexts = []

    for section_title in ["_short_summary", "_deprecation", "_extended_summary"]:
        if section_title in sections:
            section_fulltexts.append(sections.pop(section_title) + "\n")

    for section_title in NUMPYDOC_SECTION_TITLES:
        try:
            section_content = sections.pop(section_title)
        except KeyError:
            continue

        section_fulltexts.append(
            f"{section_title}\n{'-' * len(section_title)}\n{section_content}\n"
        )

    # Assemble the final docstring
    doc = "\n".join(section_fulltexts) + "\n"
    return doc
