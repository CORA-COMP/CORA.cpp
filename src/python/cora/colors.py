"""CORA's colors, as MATLAB CORA's `CORAcolor`, for matplotlib and anything else that takes RGB."""
import numpy as np

# The special colors, and the default palette (`CORA:color1` .. `CORA:color7`).
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
        from matplotlib.colors import to_rgb
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
