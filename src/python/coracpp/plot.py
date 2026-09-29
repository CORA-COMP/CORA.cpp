"""Plotting in the style of MATLAB CORA: `plot(obj, dims)` draws what it is given, in CORA's
colors, on the current axes (or `ax`).

    coracpp.plot(R)                        a reachable set from `coracpp.reach`
    coracpp.plot((c, G))                   a zonotope, given by its center and generators
    coracpp.plot(simulation)               trajectories from `coracpp.simulate`, (time points, n, N)
    coracpp.plot(points)                   points from `coracpp.randPoint`, (n, N)
    coracpp.plot(spec)                     a `coracpp.Specification`: the region it forbids

`dims` picks the two dimensions to show (a zonotope is projected onto them by keeping those
rows of `c` and `G`). Pass `label=` to name what is drawn, for `ax.legend()`, and matplotlib
style arguments (`facecolor`, `edgecolor`, `alpha`, ...) to change how. Every function also
exists under its own name (`plot_reach`, `plot_zonotope`, ...). matplotlib is imported when
something is drawn, so `import coracpp` does not need it.
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


def plot_zonotope(c, G, dims=(0, 1), ax=None, label=None, **style):
    """Draws the zonotope (c, G), projected onto `dims`, as an outline (pass `facecolor` to
    fill it); returns the patch."""
    ax = _axes(ax)
    dims = list(dims)
    style.setdefault("facecolor", "none")
    style.setdefault("edgecolor", CORAcolor("CORA:blue"))
    patch = _polygon(zonotope_vertices(to_numpy(c)[dims], to_numpy(G)[dims]), label=label, **style)
    ax.add_patch(patch)
    ax.autoscale_view()
    return patch


def plot_interval(inf, sup, dims=(0, 1), ax=None, label=None, **style):
    """Draws the box [inf, sup], projected onto `dims`, as an outline."""
    inf, sup = to_numpy(inf).reshape(-1), to_numpy(sup).reshape(-1)
    return plot_zonotope((inf + sup) / 2, np.diag((sup - inf) / 2), dims, ax, label, **style)


def plot_initial_set(c, G, dims=(0, 1), ax=None, label=None):
    """The initial set: white with a black outline, as in CORA."""
    return plot_zonotope(c, G, dims, ax, label, facecolor=CORAcolor("CORA:initialSet"),
                         edgecolor="black", linewidth=1.0, zorder=3)


def plot_reach(reach, dims=(0, 1), ax=None, label=None, time_points=False, step=1,
               num_colors=1, cidx=1):
    """Draws the time-interval sets of a `coracpp.reach` result in CORA's reachable-set blue,
    and optionally the time-point sets as dotted outlines. `reach` must hold one set (no batch
    dimensions); `step` skips sets; several results can be told apart with `num_colors` and
    `cidx`, as in `CORAcolor("CORA:reachSet", num_colors, cidx)`."""
    ax = _axes(ax)
    color = CORAcolor("CORA:reachSet", num_colors, cidx)
    for k in range(0, reach.timeInt_c.shape[0], step):
        plot_zonotope(reach.timeInt_c[k], reach.timeInt_G[k], dims, ax,
                      label=label if k == 0 else None, facecolor=color, edgecolor="none", zorder=1)
    if time_points:
        for k in range(0, reach.timePoint_c.shape[0], step):
            plot_zonotope(reach.timePoint_c[k], reach.timePoint_G[k], dims, ax,
                          edgecolor=CORAcolor("CORA:reachSet:dark"), linewidth=0.5,
                          linestyle=":", zorder=2)
    ax.set_aspect("equal", adjustable="box")


def plot_simulation(simulation, dims=(0, 1), ax=None, label=None, markers=True):
    """Draws simulated trajectories in CORA's simulation color. `simulation` is what
    `coracpp.simulate` returns for one system, `(time points, n, N)`: N trajectories, with a
    dot where each starts."""
    ax = _axes(ax)
    x = to_numpy(simulation)
    d0, d1 = dims
    color = CORAcolor("CORA:simulations")
    lines = ax.plot(x[:, d0, :], x[:, d1, :], color=color, linewidth=0.6, zorder=4)
    if label is not None:
        lines[0].set_label(label)
    if markers:
        ax.plot(x[0, d0, :], x[0, d1, :], ".", color=color, markersize=3, zorder=4)
    ax.autoscale_view()
    return lines


def plot_points(points, dims=(0, 1), ax=None, label=None, **style):
    """Draws points, `(n, N)` as `coracpp.randPoint` returns them."""
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
    """Shades the region a `coracpp.Specification` forbids, on the axes' current limits, in
    CORA's unsafe colors: for a safe set the outside of each halfspace, for an unsafe set the
    halfspace itself. Draw the sets first, or set the limits, so that the region has an extent."""
    ax = _axes(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    patches = []
    for a, b in spec.halfspaces:
        # a.x <= b is the halfspace; the safe set forbids a.x > b, the unsafe set forbids a.x <= b.
        sign = 1.0 if spec.type == "safeSet" else -1.0
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
    """CORA's `plot`: draws a reachable set, a zonotope `(c, G)`, simulations `(time points,
    n, N)`, points `(n, N)`, or a specification, by what it is."""
    if hasattr(obj, "timeInt_c"):
        return plot_reach(obj, dims, ax, label, **style)
    if hasattr(obj, "halfspaces"):
        return plot_specification(obj, dims, ax, label)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return plot_zonotope(obj[0], obj[1], dims, ax, label, **style)
    array = to_numpy(obj)
    if array.ndim == 3:
        return plot_simulation(array, dims, ax, label, **style)
    if array.ndim == 2:
        return plot_points(array, dims, ax, label, **style)
    raise TypeError(f"coracpp.plot does not know how to draw a {type(obj).__name__}")
