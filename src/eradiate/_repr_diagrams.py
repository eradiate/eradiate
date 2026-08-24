"""SVG diagrams for GridCoords/SceneGeometry HTML reprs."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pint

from .units import unit_registry as ureg


def _attrib(attrib: dict) -> dict[str, str]:
    """Translate Python-friendly (underscored) attribute names to their SVG/CSS (hyphenated) form and stringify values, dropping ``None`` entries."""
    return {k.replace("_", "-"): str(v) for k, v in attrib.items() if v is not None}


def svg(width: float, height: float, *children: ET.Element, **attrib) -> ET.Element:
    """Build an <svg> root sized to (width, height) with a matching viewBox."""
    el = ET.Element(
        "svg",
        _attrib(
            {
                "width": f"{width}",
                "height": f"{height}",
                "viewBox": f"0 0 {width} {height}",
                "xmlns": "http://www.w3.org/2000/svg",
                **attrib,
            }
        ),
    )
    el.extend(children)
    return el


def g(*children: ET.Element, **attrib) -> ET.Element:
    """Build a <g> group element."""
    el = ET.Element("g", _attrib(attrib))
    el.extend(children)
    return el


def line(x1: float, y1: float, x2: float, y2: float, **attrib) -> ET.Element:
    """Build a <line> element."""
    return ET.Element(
        "line",
        _attrib({"x1": f"{x1}", "y1": f"{y1}", "x2": f"{x2}", "y2": f"{y2}", **attrib}),
    )


def rect(x: float, y: float, width: float, height: float, **attrib) -> ET.Element:
    """Build a <rect> element."""
    return ET.Element(
        "rect",
        _attrib(
            {
                "x": f"{x}",
                "y": f"{y}",
                "width": f"{width}",
                "height": f"{height}",
                **attrib,
            }
        ),
    )


def circle(cx: float, cy: float, r: float, **attrib) -> ET.Element:
    """Build a <circle> element."""
    return ET.Element(
        "circle", _attrib({"cx": f"{cx}", "cy": f"{cy}", "r": f"{r}", **attrib})
    )


def ellipse(cx: float, cy: float, rx: float, ry: float, **attrib) -> ET.Element:
    """Build an <ellipse> element."""
    return ET.Element(
        "ellipse",
        _attrib({"cx": f"{cx}", "cy": f"{cy}", "rx": f"{rx}", "ry": f"{ry}", **attrib}),
    )


def polygon(points: list[tuple[float, float]], **attrib) -> ET.Element:
    """Build a <polygon> element from a sequence of (x, y) points."""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return ET.Element("polygon", _attrib({"points": pts, **attrib}))


def path(d: str, **attrib) -> ET.Element:
    """Build a <path> element with the given path data."""
    return ET.Element("path", _attrib({"d": d, **attrib}))


def text(x: float, y: float, content: str, **attrib) -> ET.Element:
    """Build a <text> element."""
    el = ET.Element("text", _attrib({"x": f"{x}", "y": f"{y}", **attrib}))
    el.text = content
    return el


def el(
    tag: str, *children: ET.Element, text: str | None = None, **attrib
) -> ET.Element:
    """Build a generic (typically non-SVG, e.g. HTML) element with optional text content and children."""
    node = ET.Element(tag, _attrib(attrib))
    node.text = text
    node.extend(children)
    return node


def tostring(node: ET.Element) -> str:
    """Serialize an element tree to an XML string."""
    return ET.tostring(node, encoding="unicode")


def _division_fractions(
    n_cells: int, *, max_shown: int = 5, gap_scale: float = 2.0
) -> tuple[list[float], tuple[float, float] | None]:
    """Division fractions for up to *max_shown* grid segments, plus the compressed-gap range (*gap_scale* times as wide as a regular cell) when *n_cells* exceeds *max_shown*."""
    if n_cells <= 1:
        return [], None
    if n_cells <= max_shown:
        return [i / n_cells for i in range(1, n_cells)], None

    n_head = n_tail = 2
    unit = 1.0 / (n_head + n_tail + gap_scale)
    lines = [
        unit,
        2 * unit,
        (2 + gap_scale) * unit,
        (3 + gap_scale) * unit,
    ]
    gap = (2 * unit, (2 + gap_scale) * unit)
    return lines, gap


_AXIS_SCALES = (0.55, 1.0, 1.6)  # (small, medium, large) edge-length multipliers


def _axis_scales(
    span_x: pint.Quantity,
    span_y: pint.Quantity,
    span_z: pint.Quantity,
    scales: tuple[float, float, float] = _AXIS_SCALES,
) -> tuple[float, float, float]:
    """Rank each axis by physical extent into one of 3 predefined *scales* (small/medium/large, ties share a scale, non-comparable axes default to medium)."""
    spans = [span_x, span_y, span_z]
    small, medium, large = scales

    def _cmp(i: int, j: int) -> int | None:
        try:
            if spans[i] < spans[j]:
                return -1
            if spans[i] > spans[j]:
                return 1
            return 0
        except pint.DimensionalityError:
            return None

    result = [medium, medium, medium]
    for i in range(3):
        others = [j for j in range(3) if j != i and _cmp(i, j) is not None]
        if not others:
            continue
        is_min = all(_cmp(i, j) <= 0 for j in others)
        is_max = all(_cmp(i, j) >= 0 for j in others)
        if is_min and not is_max:
            result[i] = small
        elif is_max and not is_min:
            result[i] = large
    return tuple(result)


def _format_extent(lo: pint.Quantity, hi: pint.Quantity) -> str:
    return f"{lo:~P.4g} … {hi:~P.4g}"


def _visual_fraction(
    ratio: float, *, floor: float = 0.15, log_floor_ratio: float = 1e-6
) -> float:
    """Map a linear size ratio in (0, 1] to a visual fraction in [*floor*, 1] on a log scale."""
    if ratio >= 1.0:
        return 1.0
    if ratio <= log_floor_ratio:
        return floor
    t = math.log10(ratio) / math.log10(log_floor_ratio)  # 0 at ratio=1, 1 at floor
    return floor + (1.0 - t) * (1.0 - floor)


def _add(
    p: tuple[float, float], v: tuple[float, float], f: float = 1.0
) -> tuple[float, float]:
    return (p[0] + v[0] * f, p[1] + v[1] * f)


def _pt(
    origin: tuple[float, float],
    va: tuple[float, float],
    fa: float,
    vb: tuple[float, float],
    fb: float,
) -> tuple[float, float]:
    return _add(_add(origin, va, fa), vb, fb)


def _cuboid_vertices(
    p0: tuple[float, float],
    ux: tuple[float, float],
    uy: tuple[float, float],
    uz: tuple[float, float],
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """The 6 remaining corners of a parallelepiped from its front-bottom-left corner and 3 edge vectors."""
    p1 = _add(p0, ux)
    p3 = _add(p0, uz)
    p2 = _add(p1, uz)
    p1b = _add(p1, uy)
    p3b = _add(p3, uy)
    p2b = _add(p2, uy)
    return p1, p2, p3, p1b, p2b, p3b


def _cuboid_faces(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p1b: tuple[float, float],
    p2b: tuple[float, float],
    p3b: tuple[float, float],
    *,
    stroke_width: float = 1.2,
) -> list[ET.Element]:
    """The cuboid's 3 visible (top, right, front) faces as filled, shaded polygons."""
    stroke = "#1f3a52"
    return [
        polygon(
            [p3, p2, p2b, p3b], fill="#a9c8e0", stroke=stroke, stroke_width=stroke_width
        ),
        polygon(
            [p1, p2, p2b, p1b], fill="#2f5678", stroke=stroke, stroke_width=stroke_width
        ),
        polygon(
            [p0, p1, p2, p3], fill="#4c78a8", stroke=stroke, stroke_width=stroke_width
        ),
    ]


def cuboid(
    n_x: int,
    n_y: int,
    n_z: int,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
) -> ET.Element:
    """Isometric cuboid <svg> element with the cell grid drawn on its 3 visible faces and X/Y/Z edges labelled with their cell count."""
    base_w, base_h = 106.6, 94.9  # front face size (X width, Z height) at scale 1
    base_uy = (40.3, -27.3)  # depth direction/length (Y) at scale 1, screen space

    W = base_w * scale_x
    H = base_h * scale_z
    uy = (base_uy[0] * scale_y, base_uy[1] * scale_y)
    ux = (W, 0.0)
    uz = (0.0, -H)

    # Margins are fixed regardless of scale and the canvas grows/shrinks
    # around them instead, so long axis labels (e.g. "z (n=2000000)") never
    # clip.
    left_margin, right_margin = 161.2, 88.4
    top_pad, bottom_margin = 10.4, 46.8
    p0 = (left_margin, top_pad + abs(uy[1]) + H)
    canvas_w = left_margin + W + uy[0] + right_margin
    canvas_h = top_pad + abs(uy[1]) + H + bottom_margin

    p1, p2, p3, p1b, p2b, p3b = _cuboid_vertices(p0, ux, uy, uz)
    faces = _cuboid_faces(p0, p1, p2, p3, p1b, p2b, p3b)

    x_lines, x_gap = _division_fractions(n_x)
    y_lines, y_gap = _division_fractions(n_y)
    z_lines, z_gap = _division_fractions(n_z)

    def _gridline(a: tuple[float, float], b: tuple[float, float]) -> ET.Element:
        return line(*a, *b, stroke="white", stroke_opacity=0.65, stroke_width=0.8)

    grid_lines = []
    # Front face (X-Z): X divisions run bottom-to-top, Z divisions left-to-right.
    for fx in x_lines:
        grid_lines.append(_gridline(_pt(p0, ux, fx, uz, 0), _pt(p0, ux, fx, uz, 1)))
    for fz in z_lines:
        grid_lines.append(_gridline(_pt(p0, uz, fz, ux, 0), _pt(p0, uz, fz, ux, 1)))
    # Top face (X-Y): X divisions continue into depth, Y divisions run across.
    for fx in x_lines:
        grid_lines.append(_gridline(_pt(p3, ux, fx, uy, 0), _pt(p3, ux, fx, uy, 1)))
    for fy in y_lines:
        grid_lines.append(_gridline(_pt(p3, uy, fy, ux, 0), _pt(p3, uy, fy, ux, 1)))
    # Right face (Z-Y): Z divisions continue into depth, Y divisions run across.
    for fz in z_lines:
        grid_lines.append(_gridline(_pt(p1, uz, fz, uy, 0), _pt(p1, uz, fz, uy, 1)))
    for fy in y_lines:
        grid_lines.append(_gridline(_pt(p1, uy, fy, uz, 0), _pt(p1, uy, fy, uz, 1)))

    def _ellipsis_dots(
        origin: tuple[float, float],
        axis_vec: tuple[float, float],
        axis_frac: float,
        other_vec: tuple[float, float],
        other_frac: float,
    ) -> list[ET.Element]:
        cx, cy = _pt(origin, axis_vec, axis_frac, other_vec, other_frac)
        norm = math.hypot(*axis_vec)
        if norm == 0:
            return []
        dirx, diry = axis_vec[0] / norm, axis_vec[1] / norm
        return [
            circle(
                cx + dirx * k * 4.03,
                cy + diry * k * 4.03,
                1.43,
                fill="#16283a",
            )
            for k in (-1, 0, 1)
        ]

    ellipses = []
    if x_gap:
        ellipses += _ellipsis_dots(p0, ux, sum(x_gap) / 2, uz, 0.22)
    if z_gap:
        ellipses += _ellipsis_dots(p0, uz, sum(z_gap) / 2, ux, 0.8)
    if y_gap:
        ellipses += _ellipsis_dots(p3, uy, sum(y_gap) / 2, ux, 0.22)

    label_attrib = {"font_family": "sans-serif", "font_size": "13.65px", "fill": "#555"}
    labels = [
        text(
            (p0[0] + p1[0]) / 2,
            p0[1] + 32.5,
            f"x (n={n_x})",
            text_anchor="middle",
            **label_attrib,
        ),
        text(
            p0[0] - 23.4,
            (p0[1] + p3[1]) / 2,
            f"z (n={n_z})",
            text_anchor="end",
            **label_attrib,
        ),
        text(
            (p1[0] + p1b[0]) / 2 + 28.6,
            (p1[1] + p1b[1]) / 2 + 16.9,
            f"y (n={n_y})",
            text_anchor="middle",
            **label_attrib,
        ),
    ]

    return svg(
        canvas_w,
        canvas_h,
        *faces,
        *grid_lines,
        *ellipses,
        *labels,
        style="vertical-align:middle",
    )


def nested_cuboid(
    domain_width: pint.Quantity,
    grid_width: pint.Quantity,
    grid_length: pint.Quantity,
    ground: pint.Quantity,
    toa: pint.Quantity,
) -> ET.Element:
    """Dashed outer wireframe cuboid (the geometry's full domain) with the grid's footprint drawn as a smaller solid cuboid nested inside."""
    frac_w = _visual_fraction(
        float((grid_width / domain_width).m_as(ureg.dimensionless))
    )
    frac_l = _visual_fraction(
        float((grid_length / domain_width).m_as(ureg.dimensionless))
    )

    W, H = 169.0, 104.0
    uy = (52.0, -33.8)
    ux = (W, 0.0)
    uz = (0.0, -H)

    left_margin, right_margin = 78.0, 26.0
    top_pad, bottom_margin = 15.6, 44.2
    p0 = (left_margin, top_pad + abs(uy[1]) + H)
    canvas_w = left_margin + W + uy[0] + right_margin
    canvas_h = top_pad + abs(uy[1]) + H + bottom_margin

    p1, p2, p3, p1b, p2b, p3b = _cuboid_vertices(p0, ux, uy, uz)

    stroke = "#9aa5b1"
    outer = [
        line(*a, *b, stroke=stroke, stroke_width=1.2, stroke_dasharray="4,3")
        for a, b in (
            (p0, p1),
            (p1, p2),
            (p2, p3),
            (p3, p0),
            (p3, p3b),
            (p2, p2b),
            (p1, p1b),
            (p0, _add(p0, uy)),
            (p3b, p2b),
            (p2b, p1b),
        )
    ]

    inner_w, inner_h = W * frac_w, H  # full height: grid always spans ground..toa
    inner_uy = (uy[0] * frac_l, uy[1] * frac_l)
    inner_p0 = _add(p0, ux, (1 - frac_w) / 2.0)
    inner_p0 = _add(inner_p0, uy, (1 - frac_l) / 2.0)
    iux = (inner_w, 0.0)
    iuz = (0.0, -inner_h)
    ip1, ip2, ip3, ip1b, ip2b, ip3b = _cuboid_vertices(inner_p0, iux, inner_uy, iuz)
    inner = _cuboid_faces(inner_p0, ip1, ip2, ip3, ip1b, ip2b, ip3b, stroke_width=1)

    label_attrib = {"font_family": "sans-serif", "font_size": "13.65px", "fill": "#555"}
    labels = [
        text(
            (p0[0] + p1[0]) / 2,
            p0[1] + 32.5,
            f"domain width {domain_width:~P.3g}",
            text_anchor="middle",
            **label_attrib,
        ),
        text(
            p0[0] - 10.4,
            p0[1] + 5.2,
            f"{ground:~P.4g}",
            text_anchor="end",
            **label_attrib,
        ),
        text(
            p3[0] - 10.4,
            p3[1] + 5.2,
            f"{toa:~P.4g}",
            text_anchor="end",
            **label_attrib,
        ),
    ]

    diagram = svg(
        canvas_w, canvas_h, *outer, *inner, *labels, style="vertical-align:middle"
    )
    caption = el(
        "div",
        text=f"grid footprint: {grid_width:~P.3g} × {grid_length:~P.3g}",
        style="font-family:sans-serif;font-size:13.65px;color:#555",
    )
    return el("div", diagram, caption)


def spherical_extent(
    az_min: float,
    az_max: float,
    colat_min: float,
    colat_max: float,
    ground: pint.Quantity,
    toa: pint.Quantity,
) -> ET.Element:
    """Polar plot (angle=azimuth, radius=colatitude) of the grid's angular coverage, plus an altitude bar."""
    cx, cy, R = 109.2, 109.2, 93.6

    def _polar(az_deg: float, colat_deg: float) -> tuple[float, float]:
        r = R * (colat_deg / 180.0)
        a = math.radians(az_deg - 90.0)
        return (cx + r * math.cos(a), cy + r * math.sin(a))

    fill = "#4c78a8"
    stroke = "#1f3a52"
    label_attrib = {"font_family": "sans-serif", "font_size": "13.65px", "fill": "#555"}

    wedge_children = [
        circle(
            cx,
            cy,
            R,
            fill="none",
            stroke="#c3ccd4",
            stroke_width=1,
            stroke_dasharray="3,3",
        ),
        circle(
            cx,
            cy,
            R * 0.5,
            fill="none",
            stroke="#e3e8ec",
            stroke_width=0.8,
            stroke_dasharray="2,3",
        ),
        circle(cx, cy, 1.82, fill="#9aa5b1"),
    ]

    full_azimuth = (az_max - az_min) >= 359.999
    from_pole = colat_min <= 1e-3

    if full_azimuth and from_pole:
        r = R * (colat_max / 180.0)
        wedge_children.append(
            circle(
                cx, cy, r, fill=fill, fill_opacity=0.8, stroke=stroke, stroke_width=1.2
            )
        )
    elif full_azimuth:
        r0 = R * (colat_min / 180.0)
        r1 = R * (colat_max / 180.0)
        wedge_children += [
            circle(
                cx, cy, r1, fill=fill, fill_opacity=0.8, stroke=stroke, stroke_width=1.2
            ),
            circle(cx, cy, r0, fill="white", stroke=stroke, stroke_width=1.2),
        ]
    else:
        r0 = R * (colat_min / 180.0)
        r1 = R * (colat_max / 180.0)
        p_in0 = _polar(az_min, colat_min)
        p_out0 = _polar(az_min, colat_max)
        p_out1 = _polar(az_max, colat_max)
        p_in1 = _polar(az_max, colat_min)
        large_arc = 1 if (az_max - az_min) > 180 else 0
        d = (
            f"M{p_in0[0]},{p_in0[1]} "
            f"L{p_out0[0]},{p_out0[1]} "
            f"A{r1},{r1} 0 {large_arc} 1 {p_out1[0]},{p_out1[1]} "
            f"L{p_in1[0]},{p_in1[1]} "
            f"A{r0},{r0} 0 {large_arc} 0 {p_in0[0]},{p_in0[1]} Z"
        )
        wedge_children.append(
            path(d, fill=fill, fill_opacity=0.8, stroke=stroke, stroke_width=1.2)
        )

    wedge_children += [
        text(
            cx,
            cy + R + 26.0,
            f"azimuth {az_min:g}-{az_max:g} deg",
            text_anchor="middle",
            **label_attrib,
        ),
        text(
            cx,
            cy + R + 44.2,
            f"colatitude {colat_min:g}-{colat_max:g} deg",
            text_anchor="middle",
            **label_attrib,
        ),
    ]
    wedge_svg = svg(
        2 * cx, cy + R + 57.2, *wedge_children, style="vertical-align:middle"
    )

    bar_x, bar_w = 15.6, 20.8
    bar_top, bar_bot = 18.2, cy + R - 26.0
    bar_h = bar_bot - bar_top
    bar_canvas_w, bar_canvas_h = 83.2, cy + R + 57.2
    bar_mid_x, bar_mid_y = bar_x + bar_w / 2, (bar_top + bar_bot) / 2
    bar_children = [
        rect(
            bar_x,
            bar_top,
            bar_w,
            bar_h,
            rx=2,
            fill="#a9c8e0",
            stroke=stroke,
            stroke_width=1,
        ),
        text(
            bar_mid_x,
            bar_top - 7.8,
            f"{toa:~P.4g}",
            text_anchor="middle",
            **label_attrib,
        ),
        text(
            bar_mid_x,
            bar_bot + 20.8,
            f"{ground:~P.4g}",
            text_anchor="middle",
            **label_attrib,
        ),
        text(
            bar_mid_x,
            bar_mid_y,
            "altitude",
            text_anchor="middle",
            transform=f"rotate(-90 {bar_mid_x} {bar_mid_y})",
            **label_attrib,
        ),
    ]
    bar_svg = svg(
        bar_canvas_w, bar_canvas_h, *bar_children, style="vertical-align:middle"
    )

    return el(
        "div",
        el("div", wedge_svg),
        el("div", bar_svg),
        style="display:flex;align-items:flex-start;gap:8px",
    )


def grid_repr_html(grid) -> str | None:
    from .grid import PlaneParallelGridCoords, SphericalShellGridCoords

    # Never let a display helper break notebook rendering: fall back to
    # the plain-text repr if anything goes wrong.
    try:
        if isinstance(grid, PlaneParallelGridCoords):
            axes = [
                ("x (width)", grid.edges_x),
                ("y (length)", grid.edges_y),
                ("z (altitude)", grid.levels),
            ]
        elif isinstance(grid, SphericalShellGridCoords):
            axes = [
                ("azimuth", grid.azimuths),
                ("colatitude", grid.colatitudes),
                ("z (altitude)", grid.levels),
            ]
        else:
            return None

        n_cells = [len(edges) - 1 for _, edges in axes]
        spans = [abs(edges[-1] - edges[0]) for _, edges in axes]
        scale_x, scale_y, scale_z = _axis_scales(*spans)
        diagram = cuboid(*n_cells, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)

        rows = [
            el(
                "tr",
                el(
                    "td",
                    text=label,
                    style="padding:2px 8px 2px 0;white-space:nowrap;font-weight:600",
                ),
                el(
                    "td",
                    text=_format_extent(edges[0], edges[-1]),
                    style="padding:2px 8px 2px 0;white-space:nowrap;font-family:monospace",
                ),
                el(
                    "td",
                    text=str(n),
                    style="padding:2px 0;text-align:right;white-space:nowrap",
                ),
            )
            for (label, edges), n in zip(axes, n_cells)
        ]

        table = el(
            "table",
            el(
                "thead",
                el(
                    "tr",
                    el("th", text="axis", style="text-align:left;padding:0 8px 2px 0"),
                    el(
                        "th", text="extent", style="text-align:left;padding:0 8px 2px 0"
                    ),
                    el("th", text="cells", style="text-align:right;padding:0 0 2px 0"),
                ),
            ),
            el("tbody", *rows),
            style="border-collapse:collapse",
        )

        root = el(
            "div",
            el(
                "div",
                text=type(grid).__name__,
                style="font-weight:600;margin-bottom:4px",
            ),
            el(
                "div",
                el("div", diagram),
                table,
                style="display:flex;align-items:center;gap:12px",
            ),
            style="font-size:1.17em",
        )
        return tostring(root)
    except Exception:  # noqa: BLE001 — repr helper must never raise
        return None


def geometry_repr_html(geometry) -> str | None:
    from .scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry

    # Never let a display helper break notebook rendering: fall back to
    # the plain-text repr if anything goes wrong.
    try:
        if isinstance(geometry, PlaneParallelGeometry):
            diagram = nested_cuboid(
                geometry.width,
                geometry.grid.total_width,
                geometry.grid.total_length,
                geometry.ground_altitude,
                geometry.toa_altitude,
            )
        elif isinstance(geometry, SphericalShellGeometry):
            grid = geometry.grid
            diagram = spherical_extent(
                grid.azimuths[0].m_as(ureg.deg),
                grid.azimuths[-1].m_as(ureg.deg),
                grid.colatitudes[0].m_as(ureg.deg),
                grid.colatitudes[-1].m_as(ureg.deg),
                geometry.ground_altitude,
                geometry.toa_altitude,
            )
        else:
            return None

        root = el(
            "div",
            el(
                "div",
                text=type(geometry).__name__,
                style="font-weight:600;margin-bottom:4px",
            ),
            el("div", diagram),
            style="font-size:1.17em",
        )
        return tostring(root)
    except Exception:  # noqa: BLE001 — repr helper must never raise
        return None
