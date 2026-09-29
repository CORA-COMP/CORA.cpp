"""Basic plotting of coracpp results with matplotlib, in CORA's colors.

Reachable sets, the initial set, simulations and the unsafe region are drawn in the colors
of MATLAB CORA's `CORAcolor`. A zonotope is drawn as its polygon; a higher-dimensional one is
projected onto two of its dimensions first, which for a zonotope is just keeping those rows of
`c` and `G`.
"""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

# CORAcolor: the special colors, and the default palette (`CORA:color1` .. `CORA:color7`).
_SPECIAL = {
    "initialSet": (1.0, 1.0, 1.0),
    "finalSet": (0.9, 0.9, 0.9),
    "simulations": (0.0, 0.0, 0.0),
    "unsafe": (0.9451, 0.5529, 0.5686),
    "unsafeLight": (0.9059, 0.7373, 0.7373),
    "safe": (0.4706, 0.7725, 0.4980),
    "invariant": (0.4706, 0.7725, 0.4980),
    "highlight1": (1.0, 0.6824, 0.2980),
    "highlight2": (0.6235, 0.7294, 0.2118),
}
_PALETTE = [
    ("blue", (0.0, 0.4470, 0.7410)),
    ("red", (0.8500, 0.3250, 0.0980)),
    ("yellow", (0.9290, 0.6940, 0.1250)),
    ("purple", (0.4940, 0.1840, 0.5560)),
    ("green", (0.4660, 0.6740, 0.1880)),
    ("light-blue", (0.3010, 0.7450, 0.9330)),
    ("dark-red", (0.6350, 0.0780, 0.1840)),
]
_REACH_MAIN = np.array((0.2706, 0.5882, 1.0000))
_REACH_WORSE = np.array((0.6902, 0.8235, 1.0000))


def CORAcolor(identifier, num_colors=1, cidx=1, alpha=0.2):
    """The RGB triple of a CORA color, as MATLAB CORA's `CORAcolor`.

    `identifier` is "CORA:reachSet", "CORA:initialSet", "CORA:finalSet", "CORA:simulations",
    "CORA:unsafe", "CORA:unsafeLight", "CORA:safe", "CORA:invariant", "CORA:highlight1",
    "CORA:highlight2", "CORA:next", a palette name ("CORA:blue", "CORA:red", "CORA:yellow",
    "CORA:purple", "CORA:green", "CORA:light-blue", "CORA:dark-red") or its number
    ("CORA:color3", or just 3). A postfix ":light" or ":dark" mixes the color with white or
    black by `alpha`. "CORA:reachSet" shades from light to full blue over `num_colors`
    reachable sets, `cidx` of them (1 is the lightest)."""
    if isinstance(identifier, int):
        return CORAcolor(f"CORA:color{identifier}")
    parts = identifier.split(":")
    if len(parts) < 2 or parts[0] != "CORA":
        raise ValueError(f"not a CORA color: {identifier}")
    name, variant = parts[1], parts[2] if len(parts) > 2 else "none"

    if name == "reachSet":
        if cidx > num_colors:
            raise ValueError("the color index must not be larger than the number of colors")
        if cidx == num_colors:
            color = _REACH_MAIN
        elif cidx == 1:
            color = _REACH_WORSE
        else:
            color = _REACH_WORSE + (_REACH_MAIN - _REACH_WORSE) * ((cidx - 1) / (num_colors - 1))
    elif name == "next":
        # CORA takes the next color of the axes' order; matplotlib's cycle is indexed by `cidx`.
        color = np.array(to_rgb(f"C{cidx - 1}"))
    elif name in _SPECIAL:
        color = np.array(_SPECIAL[name])
    else:
        by_name = {label: rgb for label, rgb in _PALETTE}
        by_number = {f"color{i + 1}": rgb for i, (_, rgb) in enumerate(_PALETTE)}
        if name not in by_name and name not in by_number:
            raise ValueError(f"not a CORA color: {identifier}")
        color = np.array(by_name.get(name, by_number.get(name)))

    if variant == "light":
        color = 1.0 * (1 - alpha) + color * alpha
    elif variant == "dark":
        color = 0.0 * (1 - alpha) + color * alpha
    elif variant != "none":
        raise ValueError(f"unknown color variant: {variant}")
    return tuple(float(v) for v in color)


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


def plot_initial_set(ax, c, G, dims=(0, 1)):
    """The initial set: white with a black outline, as in CORA."""
    return plot_zonotope(ax, c, G, dims, facecolor=CORAcolor("CORA:initialSet"), edgecolor="black",
                         linewidth=1.0, zorder=3)


def plot_reach(ax, reach, dims=(0, 1), time_points=False, step=1, num_colors=1, cidx=1):
    """Draws the time-interval sets of a `coracpp.reach` result in CORA's reachable-set blue,
    and optionally the time-point sets as dotted outlines. `reach` must hold one set (no batch
    dimensions); `step` skips sets; several results can be told apart with `num_colors` and
    `cidx`, as in `CORAcolor("CORA:reachSet", num_colors, cidx)`."""
    color = CORAcolor("CORA:reachSet", num_colors, cidx)
    for k in range(0, reach.time_int_c.shape[0], step):
        plot_zonotope(ax, reach.time_int_c[k], reach.time_int_G[k], dims, facecolor=color,
                      edgecolor="none", zorder=1)
    if time_points:
        for k in range(0, reach.time_point_c.shape[0], step):
            plot_zonotope(ax, reach.time_point_c[k], reach.time_point_G[k], dims, facecolor="none",
                          edgecolor=CORAcolor("CORA:reachSet:dark"), linewidth=0.5,
                          linestyle=":", zorder=2)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")


def plot_simulation(ax, simulation, dims=(0, 1), markers=True):
    """Draws simulated trajectories in CORA's simulation color. `simulation` is what
    `coracpp.simulate` returns for one system, `(time points, n, N)`: N trajectories, with a
    dot where each starts."""
    x = to_numpy(simulation)
    d0, d1 = dims
    color = CORAcolor("CORA:simulations")
    ax.plot(x[:, d0, :], x[:, d1, :], color=color, linewidth=0.6, zorder=4)
    if markers:
        ax.plot(x[0, d0, :], x[0, d1, :], ".", color=color, markersize=3, zorder=4)
    ax.autoscale_view()


def plot_halfspace(ax, a, b, kind="safe", dims=(0, 1)):
    """Draws the wall a.x = b of a specification on the axes' current limits and shades the
    side the reachable set must stay out of in CORA's unsafe color ("safe": a.x > b; "unsafe":
    a.x <= b)."""
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
        ax.add_patch(Polygon(np.array(polygon), closed=True, facecolor=CORAcolor("CORA:unsafeLight"),
                             edgecolor=CORAcolor("CORA:unsafe"), linewidth=1.2, zorder=0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
