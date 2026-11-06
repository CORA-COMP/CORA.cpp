"""plot - plotting in the style of MATLAB CORA: `plot(obj, dims)` draws with matplotlib

This is a thin wrapper: what is drawn, in which color and on top of what is decided by the C++
library (cora/global/plot/plot.h), the same for every language. It puts everything into a C++ Figure
and draws the layers it comes back with; the figure keeps them, so `cora.subplots` can write them
as TikZ. `dims` picks the one to three dimensions to show; the options of the core are `label`,
`color`, `face_color` and `edge_color` ("none" leaves a set open), `face_alpha`, `line_width`,
`unify`, `filled`, `set`, `time_points`, `step`, `num_colors` and `cidx`; any other keyword
(`alpha`, `zorder`, ...) goes to matplotlib. Colors are CORA identifiers ("CORA:red"), RGB
triples or matplotlib colors. Sets must hold one set, not a batch. matplotlib is imported when
something is drawn. Methods too: `Z.plot(dims)`, `R.plot(dims)`, `R.plot_over_time(dim)`.

Syntax:   cora.plot(R)                a reachable set: the ReachSet of LinearSys/NonlinearSys.reach
          cora.plot(Z)                a Zonotope or an Interval, filled, in the next CORA color
          cora.plot(Z, dims=(0, 1, 2))   the faces of a 3-D set (axes of projection="3d")
          cora.plot_over_time(R, dim)    a dimension of a ReachSet over time (or of simulations)
          cora.plot(simulation)       trajectories from simulate, (time points, n, N)
          cora.plot(points)           points from rand_point, (n, N)
          cora.plot(spec)             a Specification: the region it forbids
          cora.use_cora_colors("CORA:contDynamics")   sets as initial sets, reachable sets in blue

Authors:       Claude Sonnet 5.5
Written:       01-October-2026
Last update:   ---
Reviewed:      ---
"""
from ._cora import ContSet, Figure, ReachSet, Rng, Specification


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

_CORE_OPTIONS = ("face_alpha", "line_width", "unify", "filled", "set", "time_points", "step",
                 "num_colors", "cidx")
_ALIASES = {"facecolor": "face_color", "edgecolor": "edge_color"}


def _axes(ax, three_d):
    """The axes to draw on: `ax`, else the current ones (new, with a 3-D projection, on a figure
    that has none)."""
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    if three_d and not plt.get_fignums():
        return plt.figure().add_subplot(projection="3d")
    if three_d and not plt.gcf().axes:
        return plt.gcf().add_subplot(projection="3d")
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
    for alias, name in _ALIASES.items():
        if alias in style:
            style[name] = style.pop(alias)
    for key in ("color", "face_color", "edge_color"):
        if key not in style:
            continue
        color = style.pop(key)
        if key == "face_color" and isinstance(color, str) and color == "none":
            options["filled"] = False
        else:
            options[key] = _rgb(color)
    for key in _CORE_OPTIONS:
        if key in style:
            options[key] = style.pop(key)
    return options


def _data(obj):
    """What the core plots: a set, a reachable set, a specification or an array (torch or numpy)
    of points (n, N) or of a simulation (time points, n, N)."""
    if isinstance(obj, (ContSet, ReachSet, Specification)):
        return obj
    try:
        import numpy as np
        array = obj if hasattr(obj, "detach") else np.asarray(obj, dtype=float)
    except (TypeError, ValueError):
        array = None
    if array is not None and getattr(array, "ndim", 0) in (2, 3):
        return array
    raise TypeError(f"cora.plot does not know how to draw a {type(obj).__name__}; it draws a "
                    "ReachSet, Zonotope, Interval, Specification, simulations (time points, n, N) "
                    "or points (n, N)")


def _draw_polygons(ax, layer, style):
    """All polygons of the layer as one path, so that holes stay open and the fill is one region."""
    import numpy as np
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    paths = [Path(polygon, closed=len(polygon) >= 3) for polygon in layer["polygons"]]
    face = layer["face"]
    patch = PathPatch(
        Path.make_compound_path(*paths) if paths else Path(np.empty((0, 2))),
        facecolor=(*face, layer["face_alpha"]) if face is not None else "none",
        fill=face is not None, edgecolor=layer["edge"], linewidth=layer["line_width"],
        linestyle=":" if layer["dashed"] else "-", joinstyle="miter", zorder=layer["zorder"],
        label=layer["label"] or None)
    patch.set(**style)
    ax.add_patch(patch)
    return patch


def _draw_region(ax, layer, style):
    import numpy as np
    from matplotlib.patches import Polygon
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    region = Figure.clip_region(corners, layer["a"], layer["b"], layer["sign"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if len(region) < 3:
        return None
    patch = Polygon(region, closed=True, facecolor=layer["face"], edgecolor=layer["edge"],
                    linewidth=1.2, zorder=layer["zorder"], label=layer["label"] or None)
    patch.set(**style)
    ax.add_patch(patch)
    return patch


def _draw_surface(ax, layer, style):
    """The faces of a polytope on 3-D axes; the viewer sorts them by depth."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    face = layer["face"]
    collection = Poly3DCollection(
        layer["faces"], facecolors=(*face, layer["face_alpha"]) if face is not None else "none",
        edgecolors=layer["edge"], linewidths=layer["line_width"], zorder=layer["zorder"],
        label=layer["label"] or None)
    collection.set(**style)
    ax.add_collection3d(collection)
    return collection


def _draw(ax, layer, style):
    """One layer of the core on the axes; `style` are the matplotlib overrides."""
    kind = layer["kind"]
    if kind == "polygons":
        return _draw_polygons(ax, layer, style)
    if kind == "region":
        return _draw_region(ax, layer, style)
    if kind == "surface":
        return _draw_surface(ax, layer, style)
    x, y = layer["polygons"][0].T
    if kind == "polyline":
        (line,) = ax.plot(x, y, color=layer["edge"], linewidth=layer["line_width"],
                          zorder=layer["zorder"], label=layer["label"] or None)
    else:
        (line,) = ax.plot(x, y, ".", color=layer["edge"], markersize=2 * layer["radius"],
                          zorder=layer["zorder"], label=layer["label"] or None)
    line.set(**style)
    return line


# ============================================  MAIN  ============================================ #

def core_figure(ax):
    """The core figure of the axes: it holds every layer drawn on them, with the choices of the
    core, one per axes so that the sets take one color after another."""
    if getattr(ax, "_cora_figure", None) is None:
        ax._cora_figure = Figure()
        ax._cora_drawn = 0
        if getattr(ax, "name", "") == "3d":
            ax.view_init(elev=ax._cora_figure.elevation, azim=ax._cora_figure.azimuth - 90)
    return ax._cora_figure


def _show(ax, figure, style):
    """Draws the layers the core added since the last call; returns what was drawn."""
    layers = figure.layers()
    artists = [_draw(ax, layer, style) for layer in layers[ax._cora_drawn:]]
    ax._cora_drawn = len(layers)
    for name in ("x", "y", "z"):
        text, setter = getattr(figure, name + "label"), getattr(ax, f"set_{name}label", None)
        if text and setter and not getattr(ax, f"get_{name}label")():
            setter(text)
    if figure.equal_axes:
        ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    return artists[0] if len(artists) == 1 else artists


def plot(obj, dims=(0, 1), ax=None, label=None, **style):
    """CORA's `plot`: draws a ReachSet, a Zonotope, an Interval, simulations `(time points, n, N)`,
    points `(n, N)`, or a Specification, by what it is, on one, two or (for sets, with the keyword
    `ax` on axes of projection="3d") three dimensions `dims`; returns what it drew."""
    ax = _axes(ax, len(dims) == 3)
    options = _core_options(label, style)
    data = _data(obj)
    figure = core_figure(ax)
    figure.plot(data, [int(d) for d in dims], **options)
    return _show(ax, figure, style)


def plot_over_time(obj, dim=0, ax=None, label=None, time_step=None, **style):
    """CORA's `plotOverTime`: dimension `dim` of a ReachSet (boxes over its time intervals), or of
    simulations `(time points, n, N)` with their `time_step`, over time."""
    ax = _axes(ax, False)
    options = _core_options(label, style)
    figure = core_figure(ax)
    if isinstance(obj, ReachSet):
        figure.plot_over_time(obj, int(dim), **options)
    else:
        if time_step is None:
            raise TypeError("plot_over_time of simulations needs their time_step=...")
        figure.plot_over_time(_data(obj), int(dim), float(time_step), **options)
    return _show(ax, figure, style)


def plot_rand_point(obj, dims=(0, 1), num_points=100, rng=None, ax=None, label=None, **style):
    """CORA's `plotRandPoint`: `num_points` random points of the set as dots, e.g. over the input
    set of a network and the points it maps them to."""
    ax = _axes(ax, False)
    options = _core_options(label, style)
    figure = core_figure(ax)
    rng = Rng(0) if rng is None else rng
    figure.plot_rand_point(obj, [int(d) for d in dims], int(num_points), rng, **options)
    return _show(ax, figure, style)


# The plot functions are methods of the classes they draw: Z.plot(), R.plot_over_time(0).
def _method(function):
    def method(self, *args, **kwargs):
        return function(self, *args, **kwargs)
    method.__doc__ = function.__doc__
    method.__name__ = function.__name__
    return method


ContSet.plot = _method(plot)
ContSet.plot_rand_point = _method(plot_rand_point)
ReachSet.plot = _method(plot)
ReachSet.plot_over_time = _method(plot_over_time)
Specification.plot = _method(plot)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
