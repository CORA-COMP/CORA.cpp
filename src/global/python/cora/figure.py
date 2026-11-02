"""figure - a figure of several axes that keeps what CORA drew on each, so that it can be written as
TikZ

    fig = cora.subplots(2, 2, sharex=True)       # like plt.subplots, but one object
    cora.plot(R, dims=(0, 1), ax=fig[0, 0])
    fig.to_tikz("reach.tikz")                    # one pgfplots groupplot; fig.tikz() is the text
    fig                                          # in Jupyter: the figure as SVG, from the C++ core

`cora.subplots` returns a `Figure` that wraps the matplotlib figure (`fig.mpl`) and its axes
(`fig[row, col]`, `fig[k]`, `fig.axes`, iteration). Everything is drawn through `cora.plot`, which
records the layers the C++ core decided on in each axes; the TikZ and SVG text is written by the
core from those, in the same way for every language. Titles, axis labels and limits set with
matplotlib are taken over when the text is written. `cora.to_tikz(path)` does the same for the
current matplotlib figure, however its axes were made.
"""
from ._cora import Figure as _CoreFigure
from ._cora import FigureGrid as _CoreGrid
from .plot import core_figure


class Figure:
    """The axes of a matplotlib figure and the layers CORA drew on them."""

    def __init__(self, mpl_figure, axes, rows, cols, sharex=False, sharey=False):
        self.mpl = mpl_figure
        self.rows, self.cols = rows, cols
        self.sharex, self.sharey = sharex, sharey
        self._axes = list(axes)            # row by row

    @classmethod
    def of(cls, mpl_figure):
        """The CORA figure of a matplotlib figure: its axes in the cells of their grid."""
        axes = list(mpl_figure.axes)
        shape = [(a.get_subplotspec().get_gridspec().get_geometry()
                  if a.get_subplotspec() else (1, 1)) for a in axes]
        rows = max((s[0] for s in shape), default=1)
        cols = max((s[1] for s in shape), default=1)
        cells = [None] * (rows * cols)
        for ax in axes:
            spec = ax.get_subplotspec()
            at = (spec.rowspan.start * cols + spec.colspan.start) if spec else 0
            cells[at] = ax
        return cls(mpl_figure, [a for a in cells if a is not None], rows, cols,
                   getattr(mpl_figure, "_cora_sharex", False),
                   getattr(mpl_figure, "_cora_sharey", False))

    @property
    def axes(self):
        """The matplotlib axes, row by row."""
        return list(self._axes)

    def __getitem__(self, index):
        """The axes at (row, col), or the k-th in row order."""
        if isinstance(index, tuple):
            row, col = index
            return self._axes[(row % self.rows) * self.cols + (col % self.cols)]
        return self._axes[index]

    def __iter__(self):
        return iter(self._axes)

    def __len__(self):
        return len(self._axes)

    def core(self):
        """The C++ figure grid with everything drawn so far, and the matplotlib texts and limits."""
        grid = _CoreGrid(self.rows, self.cols)
        grid.share_x, grid.share_y = self.sharex, self.sharey
        for k, ax in enumerate(self._axes):
            figure = core_figure(ax)
            _take_over(ax, figure)
            grid.set(k // self.cols, k % self.cols, figure)
        return grid

    def tikz(self, **options):
        """The figure as pgfplots code (a groupplot for several axes): `standalone` (a document that
        compiles alone), `coramacs` (CORA's colors by name), `thin` and `thin_tolerance`, `width`,
        `height`."""
        return self.core().tikz(**options)

    def to_tikz(self, path, **options):
        """Writes the figure as pgfplots code to `path`; `external_data=True` also writes the
        coordinates to `<path without extension>_<k>.dat` tables. Options as in `tikz`."""
        self.core().save_tikz(str(path), **options)

    def svg(self):
        """The figure as SVG text, drawn by the C++ core."""
        return self.core().svg()

    def _repr_svg_(self):
        return self.svg()

    def savefig(self, *args, **kwargs):
        """matplotlib's savefig of the figure."""
        return self.mpl.savefig(*args, **kwargs)

    def show(self):
        """matplotlib's show."""
        import matplotlib.pyplot as plt
        plt.show()


def _take_over(ax, figure):
    """What matplotlib knows about the axes and the core does not: texts and fixed limits."""
    figure.title = ax.get_title() or figure.title
    for name in ("x", "y", "z"):
        getter = getattr(ax, f"get_{name}label", None)
        if getter and getter():
            setattr(figure, name + "label", getter())
        if getter and not getattr(ax, f"get_autoscale{name}_on")():
            setattr(figure, name + "lim", tuple(getattr(ax, f"get_{name}lim")()))
    if getattr(ax, "name", "") == "3d":
        figure.elevation, figure.azimuth = ax.elev, ax.azim + 90


def subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, projection=None, figsize=None, **kw):
    """`plt.subplots` for CORA: a `Figure` of nrows x ncols axes; sharing an axis shares its limits
    and, in TikZ, its labels. `projection="3d"` makes all axes three-dimensional (for sets of three
    dimensions), `projection={(row, col): "3d"}` only those cells."""
    import matplotlib.pyplot as plt
    kwargs = dict(squeeze=False, sharex=sharex, sharey=sharey, **kw)
    if figsize is not None:
        kwargs["figsize"] = figsize
    if isinstance(projection, str):
        kwargs["subplot_kw"] = {"projection": projection}
    mpl_figure, axes = plt.subplots(nrows, ncols, **kwargs)
    for (row, col), name in (projection if isinstance(projection, dict) else {}).items():
        spec = axes[row][col].get_subplotspec()
        axes[row][col].remove()
        axes[row][col] = mpl_figure.add_subplot(spec, projection=name)
    mpl_figure._cora_sharex, mpl_figure._cora_sharey = bool(sharex), bool(sharey)
    return Figure(mpl_figure, axes.ravel(), nrows, ncols, bool(sharex), bool(sharey))


def to_tikz(path, fig=None, **options):
    """Writes a matplotlib figure (the current one) as pgfplots code; see `Figure.to_tikz`."""
    import matplotlib.pyplot as plt
    Figure.of(plt.gcf() if fig is None else fig).to_tikz(path, **options)
