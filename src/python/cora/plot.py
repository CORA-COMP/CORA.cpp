"""Plotting in the style of MATLAB CORA: `plot(obj, dims)` draws what it is given, in CORA's
colors, on the current axes (or `ax`).

    cora.plot(R)                a reachable set: the Reach of LinearSys.reach
    cora.plot(Z)                a Zonotope, or an Interval
    cora.plot(simulation)       trajectories from LinearSys.simulate, (time points, n, N)
    cora.plot(points)           points from randPoint, (n, N)
    cora.plot(spec)             a Specification: the region it forbids

`dims` picks the two dimensions to show (a zonotope is projected onto them by keeping those
rows of its center and generators). Pass `label=` to name what is drawn, for `ax.legend()`, and
matplotlib style arguments (`facecolor`, `edgecolor`, `alpha`, ...) to change how. Every
function also exists under its own name (`plot_reach`, `plot_zonotope`, ...). Sets must hold
one set, not a batch. matplotlib is imported when something is drawn, so `import cora` does
not need it.
"""
import numpy as np

from .colors import CORAcolor


def _axes(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    return plt.gca()


def _polygon(vertices, **style):
    from matplotlib.patches import Polygon
    return Polygon(vertices, closed=True, **style)


def to_numpy(x):
    """A torch tensor (any device, with or without gradient) or array as a float numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def zonotope_vertices(c, G):
    """The vertices of the 2D zonotope {c + G b : |b|_inf <= 1}, counter-clockwise.

    The generators, turned into the upper half-plane and sorted by angle, are the polygon's
    edges: walk 2 g_i from the vertex c - sum(g) through all of them, and back."""
    c = to_numpy(c).reshape(2)
    G = to_numpy(G).reshape(2, -1)
    G = G[:, np.linalg.norm(G, axis=0) > 1e-14]
    if G.shape[1] == 0:
        return c[None, :]
    flip = (G[1] < 0) | ((G[1] == 0) & (G[0] < 0))
    G = np.where(flip, -G, G)
    G = G[:, np.argsort(np.arctan2(G[1], G[0]))]
    vertices = [c - G.sum(axis=1)]
    for g in G.T:
        vertices.append(vertices[-1] + 2 * g)
    for g in G.T[:-1]:
        vertices.append(vertices[-1] - 2 * g)
    return np.array(vertices)


def plot_zonotope(Z, dims=(0, 1), ax=None, label=None, **style):
    """Draws the Zonotope Z, projected onto `dims`, as an outline (pass `facecolor` to fill it);
    returns the patch."""
    ax = _axes(ax)
    dims = list(dims)
    style.setdefault("facecolor", "none")
    style.setdefault("edgecolor", CORAcolor("CORA:blue"))
    vertices = zonotope_vertices(to_numpy(Z.c)[dims], to_numpy(Z.G)[dims])
    patch = _polygon(vertices, label=label, **style)
    ax.add_patch(patch)
    ax.autoscale_view()
    return patch


def plot_interval(I, dims=(0, 1), ax=None, label=None, **style):
    """Draws the box Interval I, projected onto `dims`, as an outline."""
    inf, sup = to_numpy(I.inf).reshape(-1), to_numpy(I.sup).reshape(-1)
    c, G = (inf + sup) / 2, np.diag((sup - inf) / 2)
    ax = _axes(ax)
    style.setdefault("facecolor", "none")
    style.setdefault("edgecolor", CORAcolor("CORA:blue"))
    patch = _polygon(zonotope_vertices(c[list(dims)], G[list(dims)]), label=label, **style)
    ax.add_patch(patch)
    ax.autoscale_view()
    return patch


def plot_initial_set(Z, dims=(0, 1), ax=None, label=None):
    """The initial set: white with a black outline, as in CORA."""
    return plot_zonotope(Z, dims, ax, label, facecolor=CORAcolor("CORA:initialSet"),
                         edgecolor="black", linewidth=1.0, zorder=3)


def plot_reach(R, dims=(0, 1), ax=None, label=None, time_points=False, step=1, num_colors=1,
               cidx=1):
    """Draws the time-interval sets of a Reach in CORA's reachable-set blue, and optionally the
    time-point sets as dotted outlines. `step` skips sets; several results can be told apart with
    `num_colors` and `cidx`, as in `CORAcolor("CORA:reachSet", num_colors, cidx)`."""
    ax = _axes(ax)
    color = CORAcolor("CORA:reachSet", num_colors, cidx)
    for k in range(0, len(R.timeInt), step):
        plot_zonotope(R.timeInt[k], dims, ax, label=label if k == 0 else None, facecolor=color,
                      edgecolor="none", zorder=1)
    if time_points:
        for k in range(0, len(R.timePoint), step):
            plot_zonotope(R.timePoint[k], dims, ax, edgecolor=CORAcolor("CORA:reachSet:dark"),
                          linewidth=0.5, linestyle=":", zorder=2)
    ax.set_aspect("equal", adjustable="box")


def plot_simulation(simulation, dims=(0, 1), ax=None, label=None):
    """Draws simulated trajectories in CORA's simulation color. `simulation` is what
    `LinearSys.simulate` returns for one system, `(time points, n, N)`: N trajectories."""
    ax = _axes(ax)
    x = to_numpy(simulation)
    d0, d1 = dims
    color = CORAcolor("CORA:simulations")
    lines = ax.plot(x[:, d0, :], x[:, d1, :], color=color, linewidth=0.6, zorder=4)
    if label is not None:
        lines[0].set_label(label)
    ax.autoscale_view()
    return lines


def plot_points(points, dims=(0, 1), ax=None, label=None, **style):
    """Draws points, `(n, N)` as `randPoint` returns them."""
    ax = _axes(ax)
    p = to_numpy(points)
    style.setdefault("color", CORAcolor("CORA:simulations"))
    style.setdefault("markersize", 3)
    (line,) = ax.plot(p[dims[0]], p[dims[1]], ".", label=label, zorder=4, **style)
    ax.autoscale_view()
    return line


def _clip_to_side(corners, a, b, sign):
    """The part of the polygon `corners` where sign * (a.x - b) >= 0, edge by edge."""
    inside = sign * (corners @ a - b) >= 0
    polygon = []
    for i in range(len(corners)):
        p, q = corners[i], corners[(i + 1) % len(corners)]
        if inside[i]:
            polygon.append(p)
        if inside[i] != inside[(i + 1) % len(corners)]:
            t = (b - a @ p) / (a @ (q - p))
            polygon.append(p + t * (q - p))
    return np.array(polygon)


def plot_specification(spec, dims=(0, 1), ax=None, label=None):
    """Shades the region a Specification forbids, on the axes' current limits, in CORA's unsafe
    colors: for a safe set the outside of each halfspace, for an unsafe set the halfspace itself.
    Draw the sets first, or set the limits, so that the region has an extent."""
    ax = _axes(ax)
    # A halfspace is a.x <= b; a safe set forbids a.x > b, an unsafe set forbids a.x <= b.
    if spec.type == "safeSet":
        sign = 1.0
    elif spec.type == "unsafeSet":
        sign = -1.0
    else:
        raise ValueError(f"unknown specification type '{spec.type}'; "
                         "expected 'safeSet' or 'unsafeSet'")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    patches = []
    for a, b in spec.halfspaces:
        polygon = _clip_to_side(corners, np.asarray(a)[list(dims)], b, sign)
        if len(polygon) >= 3:
            patch = _polygon(polygon, facecolor=CORAcolor("CORA:unsafeLight"),
                             edgecolor=CORAcolor("CORA:unsafe"), linewidth=1.2, zorder=0,
                             label=label if not patches else None)
            ax.add_patch(patch)
            patches.append(patch)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return patches


def plot(obj, dims=(0, 1), ax=None, label=None, **style):
    """CORA's `plot`: draws a Reach, a Zonotope, an Interval, simulations `(time points, n, N)`,
    points `(n, N)`, or a Specification, by what it is."""
    if hasattr(obj, "timeInt"):
        return plot_reach(obj, dims, ax, label, **style)
    if hasattr(obj, "halfspaces"):
        return plot_specification(obj, dims, ax, label)
    if hasattr(obj, "G") and hasattr(obj, "c"):
        return plot_zonotope(obj, dims, ax, label, **style)
    if hasattr(obj, "inf") and hasattr(obj, "sup"):
        return plot_interval(obj, dims, ax, label, **style)
    try:
        array = to_numpy(obj)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.ndim == 3:
        return plot_simulation(array, dims, ax, label, **style)
    if array is not None and array.ndim == 2:
        return plot_points(array, dims, ax, label, **style)
    raise TypeError(f"cora.plot does not know how to draw a {type(obj).__name__}; it draws a "
                    "Reach, Zonotope, Interval, Specification, simulations (time points, n, N) "
                    "or points (n, N)")
