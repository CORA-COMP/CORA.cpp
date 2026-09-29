"""Basic plotting of coracpp results with matplotlib: zonotopes, reachable sets, halfspaces.

A zonotope is drawn as its polygon; a higher-dimensional one is projected onto two of its
dimensions first, which for a zonotope is just keeping those rows of `c` and `G`.
"""
import numpy as np
from matplotlib.patches import Polygon


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


def plot_zonotope(ax, c, G, dims=(0, 1), **style):
    """Draws the zonotope (c, G), projected onto `dims`, and returns its patch."""
    dims = list(dims)
    polygon = Polygon(zonotope_vertices(to_numpy(c)[dims], to_numpy(G)[dims]), closed=True, **style)
    ax.add_patch(polygon)
    return polygon


def plot_reach(ax, reach, dims=(0, 1), time_points=True, step=1, color="tab:blue"):
    """Draws the time-interval sets of a `coracpp.reach` result, and optionally the time-point
    sets as outlines. `reach` must hold one set (no batch dimensions); `step` skips sets."""
    for k in range(0, reach.time_int_c.shape[0], step):
        plot_zonotope(ax, reach.time_int_c[k], reach.time_int_G[k], dims,
                      facecolor=color, edgecolor=color, alpha=0.25, linewidth=0.6)
    if time_points:
        for k in range(0, reach.time_point_c.shape[0], step):
            plot_zonotope(ax, reach.time_point_c[k], reach.time_point_G[k], dims,
                          facecolor="none", edgecolor="black", linewidth=0.5, linestyle=":")
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")


def plot_halfspace(ax, a, b, kind="safe", dims=(0, 1)):
    """Draws the wall a.x = b of a specification on the axes' current limits and shades the
    side the reachable set must stay out of ("safe": a.x > b; "unsafe": a.x <= b)."""
    a = to_numpy(a).reshape(-1)[list(dims)]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    # Clip the axes' rectangle to the forbidden side, one edge at a time.
    sign = 1.0 if kind == "safe" else -1.0
    inside = sign * (corners @ a - b) >= 0
    polygon = []
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        if inside[i]:
            polygon.append(p)
        if inside[i] != inside[(i + 1) % 4]:
            t = (b - a @ p) / (a @ (q - p))
            polygon.append(p + t * (q - p))
    if len(polygon) >= 3:
        ax.add_patch(Polygon(np.array(polygon), closed=True, facecolor="tab:red", alpha=0.15,
                             edgecolor="tab:red", linewidth=1.2))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
