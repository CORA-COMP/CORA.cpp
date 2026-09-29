// plot - draws sets, reachable sets, specifications and trajectories into a figure
//
// As CORA's plot: the set is projected onto two dimensions, its vertices are computed, and the
// vertices are drawn with the plot options. This is the one place that decides how things look
// (the color scheme, the order of colors, widths, what lies on top); the SVG writer of Figure
// and the matplotlib wrapper of the Python package only draw what it decided.
//
// Syntax:   plot(X0, {0, 1});   plot(R, {0, 1}, {.label = "reachable set"});
//           plot(spec, {0, 1});   plot(x, {0, 1});   figure().save("figure.svg");
//           useCORAcolors("CORA:contDynamics");   plot(fig, R, {0, 1})  (into a figure of your own)
// See also: global/plot/figure.h, global/plot/color.h, contSet/contSet.h

#pragma once

#include "contDynamics/linearSys/linearSys.h"
#include "global/plot/figure.h"
#include "specification/specification.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

/// What a plot looks like; anything left out takes CORA's default for what is drawn.
struct PlotOptions {
    /// Legend entry; none if empty.
    std::string label = "";

    /// The color of a set: its outline and, unless `facecolor` is given, its fill; of trajectories
    /// their lines; of a reachable set its fill. A CORA color converts implicitly, e.g.
    /// `{.color = "CORA:red"}`.
    std::optional<Color> color = std::nullopt;

    /// The fill of a set, if it is to differ from its outline (which then stays black).
    std::optional<Color> facecolor = std::nullopt;

    /// The opacity of the fill of a set, 0 to 1: 0.2 for a filled set, as CORA draws it, except
    /// the white of an initial set and reachable sets, which are opaque.
    std::optional<double> faceAlpha = std::nullopt;

    /// The width of lines and outlines in pixels.
    std::optional<double> lineWidth = std::nullopt;

    /// Whether the sets of a reachable set are drawn as their union.
    bool unify = true;

    /// Whether a set or reachable set is filled; false leaves it as an outline.
    bool filled = true;

    /// Reachable sets: also draw the time-point sets as dotted outlines, and every `step`-th set
    /// only; `numColors` and `cidx` shade several results from light to full blue (with the
    /// scheme "CORA:contDynamics"), as in `CORAcolor("CORA:reachSet", numColors, cidx)`.
    bool timePoints = false;
    int step = 1, numColors = 1, cidx = 1;
};

/// Chooses the colors of everything plotted afterwards, as CORA's `useCORAcolors`:
///   "CORA:default"      a set or reachable set takes the next color of CORA's order (blue, red,
///                       yellow, purple, green, ...), filled;
///   "CORA:contDynamics" a set is an initial set (white with a black outline) and a reachable
///                       set is CORA's reachable-set blue.
/// The default is "CORA:default"; any other name is an error.
void useCORAcolors(const std::string &scheme);

/// The set S projected onto the two dimensions `dims` (0-based), into the figure `fig`.
void plot(Figure &fig, const ContSet &S, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// The time-interval sets of R, as one region.
void plot(Figure &fig, const Reach &R, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// The region a specification forbids, shaded up to the limits of the figure: the outside of
/// each halfspace of a safe set, the halfspace of an unsafe set.
void plot(Figure &fig, const Specification &spec, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// Trajectories as `LinearSys::simulate` returns them, x[k] the points (n, N) at time k: a line
/// per trajectory.
void plot(Figure &fig, const std::vector<Tensor> &x, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// Points as columns (n, N), as `randPoint` returns them.
void plot(Figure &fig, const Tensor &points, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

// The same into the current figure().
inline void plot(const ContSet &S, const std::vector<int64_t> &dims = {0, 1},
                 const PlotOptions &options = {}) {
    plot(figure(), S, dims, options);
}
inline void plot(const Reach &R, const std::vector<int64_t> &dims = {0, 1},
                 const PlotOptions &options = {}) {
    plot(figure(), R, dims, options);
}
inline void plot(const Specification &spec, const std::vector<int64_t> &dims = {0, 1},
                 const PlotOptions &options = {}) {
    plot(figure(), spec, dims, options);
}
inline void plot(const std::vector<Tensor> &x, const std::vector<int64_t> &dims = {0, 1},
                 const PlotOptions &options = {}) {
    plot(figure(), x, dims, options);
}
inline void plot(const Tensor &points, const std::vector<int64_t> &dims = {0, 1},
                 const PlotOptions &options = {}) {
    plot(figure(), points, dims, options);
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
