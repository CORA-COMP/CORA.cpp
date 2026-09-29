"""Plotting in the style of MATLAB CORA: `plot(obj, dims)` draws what it is given on the current
axes (or `ax`) with matplotlib.

    cora.plot(R)                a reachable set: the Reach of LinearSys.reach or NonlinearSys.reach
    cora.plot(Z)                a Zonotope or an Interval: filled, in the next color of CORA's
                                order (blue, red, yellow, ...); filled=False leaves it open
    cora.plot(simulation)       trajectories from simulate, (time points, n, N)
    cora.plot(points)           points from randPoint, (n, N)
    cora.plot(spec)             a Specification: the region it forbids
    cora.useCORAcolors("CORA:contDynamics")   sets as initial sets, reachable sets in blue

This is a thin wrapper: what is drawn, in which color and on top of what is decided by the C++
library (global/plot/plot.h), the same for every language. It puts everything into a C++ Figure
and draws the layers it comes back with. `dims` picks the two dimensions to show; the options of
the core are `label`, `color`, `facecolor` ("none" leaves a set open), `faceAlpha`,
`lineWidth`, `unify`, `filled`, `timePoints`, `step`, `numColors` and `cidx`; any other keyword
(`alpha`, `zorder`, ...) goes to matplotlib. Colors are CORA identifiers ("CORA:red"), RGB
triples or matplotlib colors. Sets must hold one set, not a batch. matplotlib is imported when
something is drawn.
"""
from ._cora import ContSet, Figure, Reach, Specification

_CORE_OPTIONS = ("faceAlpha", "lineWidth", "unify", "filled", "timePoints", "step", "numColors",
                 "cidx")


def _axes(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    return plt.gca()


def _rgb(color):
    """A color for the core: a CORA identifier stays, anything matplotlib knows becomes RGB."""
    if isinstance(color, str) and color.startswith("CORA:"):
        return color
    from matplotlib.colors import to_rgb
    return tuple(to_rgb(color))


def _core_options(label, style):
    """The options of the core taken out of `style`, which keeps the matplotlib ones."""
    options = {}
    if label:
        options["label"] = label
    if "color" in style:
        options["color"] = _rgb(style.pop("color"))
    if "facecolor" in style:
        facecolor = style.pop("facecolor")
        if isinstance(facecolor, str) and facecolor == "none":
            options["filled"] = False
        else:
            options["facecolor"] = _rgb(facecolor)
    for key in _CORE_OPTIONS:
        if key in style:
            options[key] = style.pop(key)
    return options


def _data(obj):
    """What the core plots: a set, a reachable set, a specification or an array (torch or numpy)
    of points (n, N) or of a simulation (time points, n, N)."""
    if isinstance(obj, (ContSet, Reach, Specification)):
        return obj
    try:
        import numpy as np
        array = obj if hasattr(obj, "detach") else np.asarray(obj, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and getattr(array, "ndim", 0) in (2, 3):
        return array
    raise TypeError(f"cora.plot does not know how to draw a {type(obj).__name__}; it draws a "
                    "Reach, Zonotope, Interval, Specification, simulations (time points, n, N) "
                    "or points (n, N)")


def _draw_polygons(ax, layer, style):
    from matplotlib.collections import PolyCollection
    face, edge = layer["face"], layer["edge"]
    # A unified set is one region: only its outline may show, not the edges between its parts.
    hide_edges = layer["unify"] and face is not None and len(layer["polygons"]) > 1
    if face is not None:
        face = (*face, layer["faceAlpha"])
    collection = PolyCollection(
        layer["polygons"], facecolors=face if face is not None else "none",
        edgecolors="none" if hide_edges else edge, linewidths=layer["lineWidth"],
        linestyles=":" if layer["dashed"] else "-", joinstyle="miter", zorder=layer["zorder"],
        label=layer["label"] or None)
    collection.set(**style)
    ax.add_collection(collection)
    return collection


def _draw_region(ax, layer, style):
    from matplotlib.patches import Polygon
    import numpy as np
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    region = Figure.clipRegion(corners, layer["a"], layer["b"], layer["sign"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if len(region) < 3:
        return None
    patch = Polygon(region, closed=True, facecolor=layer["face"], edgecolor=layer["edge"],
                    linewidth=1.2, zorder=layer["zorder"], label=layer["label"] or None)
    patch.set(**style)
    ax.add_patch(patch)
    return patch


def _draw(ax, layer, style):
    """One layer of the core on the axes; `style` are the matplotlib overrides."""
    kind = layer["kind"]
    if kind == "polygons":
        return _draw_polygons(ax, layer, style)
    if kind == "region":
        return _draw_region(ax, layer, style)
    x, y = layer["polygons"][0].T
    if kind == "polyline":
        (line,) = ax.plot(x, y, color=layer["edge"], linewidth=layer["lineWidth"],
                          zorder=layer["zorder"], label=layer["label"] or None)
    else:
        (line,) = ax.plot(x, y, ".", color=layer["edge"], markersize=2 * layer["radius"],
                          zorder=layer["zorder"], label=layer["label"] or None)
    line.set(**style)
    return line


def plot(obj, dims=(0, 1), ax=None, label=None, **style):
    """CORA's `plot`: draws a Reach, a Zonotope, an Interval, simulations `(time points, n, N)`,
    points `(n, N)`, or a Specification, by what it is; returns what it drew."""
    ax = _axes(ax)
    options = _core_options(label, style)
    data = _data(obj)
    # One figure per axes: it counts the colors given, so the sets take one color after another.
    figure = getattr(ax, "_cora_figure", None)
    if figure is None:
        figure = ax._cora_figure = Figure()
    figure.plot(data, [int(d) for d in dims], **options)
    artists = [_draw(ax, layer, style) for layer in figure.takeLayers()]
    if figure.equalAxes:
        ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    return artists[0] if len(artists) == 1 else artists
