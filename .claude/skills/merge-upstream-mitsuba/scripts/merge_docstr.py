#!/usr/bin/env python3
"""
Three-way merge of Mitsuba's generated ``include/mitsuba/python/docstr.h``.

Git's line-based merge splits this file into hunks that cut through
multi-line docstrings, so hunk-wise resolution corrupts it. This script
instead merges declaration by declaration:

* start from the upstream file (layout and content);
* keep declarations that only exist on our side, inserted in sorted order;
* keep our edits to declarations that upstream left unchanged;
* when both sides changed the same declaration, keep upstream's and report it.

Run from the ``ext/mitsuba`` repository root while the merge is in progress::

    python merge_docstr.py <base> <ours> <theirs>

where ``<base>`` is the merge base and ``<ours>``/``<theirs>`` are the merged
refs (e.g. ``main`` and ``upstream``). The result overwrites the working tree
file; stage it with ``git add`` afterwards.
"""

import argparse
import collections
import re
import subprocess
import sys

PATH = "include/mitsuba/python/docstr.h"

# A declaration ends at the closing raw-string delimiter; some declarations
# break the line after '=', and a few use plain string literals.
DECL = re.compile(
    r'static const char \*(\w+) =\s*(?:R"doc\(.*?\)doc"|"(?:[^"\\]|\\.)*")\s*;\n',
    re.S,
)


def show(ref: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{PATH}"], capture_output=True, text=True, check=True
    ).stdout


def parse(text: str) -> tuple[dict[str, str], list[re.Match]]:
    spans = list(DECL.finditer(text))
    return {m.group(1): m.group(0) for m in spans}, spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("base")
    parser.add_argument("ours")
    parser.add_argument("theirs")
    args = parser.parse_args()

    base, _ = parse(show(args.base))
    ours, _ = parse(show(args.ours))
    up_text = show(args.theirs)
    up, spans = parse(up_text)

    # Any non-blank text between declarations means the regex missed one
    gaps = [
        up_text[a.end() : b.start()]
        for a, b in zip(spans, spans[1:])
        if up_text[a.end() : b.start()].strip()
    ]
    if gaps:
        print(f"Unparsed text between declarations: {gaps[:3]}", file=sys.stderr)
        return 1

    ours_only = sorted(k for k in ours if k not in up and k not in base)
    ours_mod = [
        k
        for k in ours
        if k in up and k in base and ours[k] != base[k] and up[k] == base[k]
    ]
    both_mod = [
        k
        for k in ours
        if k in up and k in base and ours[k] != base[k] != up[k] and ours[k] != up[k]
    ]

    merged = dict(up)
    for k in ours_mod:
        merged[k] = ours[k]

    out, pos, pending = [], 0, collections.deque(ours_only)
    for m in spans:
        while pending and pending[0] < m.group(1):
            out.append(up_text[pos : m.start()])
            pos = m.start()
            out.append(ours[pending.popleft()] + "\n")
        out.append(up_text[pos : m.start()])
        out.append(merged[m.group(1)])
        pos = m.end()
    out.extend("\n" + ours[k] for k in pending)
    out.append(up_text[pos:])
    result = "".join(out)

    names = re.findall(r"static const char \*(\w+) =", result)
    dups = [k for k, n in collections.Counter(names).items() if n > 1]
    if dups:
        print(f"Duplicate declarations: {dups}", file=sys.stderr)
        return 1

    with open(PATH, "w") as f:
        f.write(result)

    print(f"upstream declarations: {len(up)}")
    print(f"kept ours-only: {len(ours_only)}")
    print(f"kept our edits: {len(ours_mod)}")
    print(f"changed on both sides (upstream kept): {both_mod}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
