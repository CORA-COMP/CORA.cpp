// plot - draws sets, reachable sets, specifications and trajectories into a figure
//
// As CORA's plot: the set is projected onto two dimensions, its vertices are computed, and the
// vertices are drawn with the plot options. This is the one place that decides how things look
// (the color scheme, the order of colors, widths, what lies on top); the SVG writer of Figure
// and the matplotlib wrapper of the Python package only draw what it decided.
//
// Syntax:   plot(X0, {0, 1});   plot(R, {0, 1}, {.label = "reachable set"});
//           plot(spec, {0, 1});   plot(x, {0, 1});   figure().save("figure.svg");
//           use_cora_colors("CORA:contDynamics");   plot(fig, R, {0, 1})  (into a figure of your own)
// See also: cora/global/plot/figure.h, cora/global/plot/color.h, cora/cont_set/cont_set.h

#pragma once

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/plot/figure.h"
#include "cora/specification/specification.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

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
    std::optional<double> face_alpha = std::nullopt;

    /// The width of lines and outlines in pixels.
    std::optional<double> line_width = std::nullopt;

    /// Whether the sets of a reachable set are drawn as their union.
    bool unify = true;

    /// Whether a set or reachable set is filled; false leaves it as an outline.
    bool filled = true;

    /// Reachable sets: also draw the time-point sets as dotted outlines, and every `step`-th set
    /// only; `num_colors` and `cidx` shade several results from light to full blue (with the
    /// scheme "CORA:contDynamics"), as in `cora_color("CORA:reachSet", num_colors, cidx)`.
    bool time_points = false;
    int step = 1, num_colors = 1, cidx = 1;
};

/// Chooses the colors of everything plotted afterwards, as CORA's `use_cora_colors`:
///   "CORA:default"      a set or reachable set takes the next color of CORA's order (blue, red,
///                       yellow, purple, green, ...), filled;
///   "CORA:contDynamics" a set is an initial set (white with a black outline) and a reachable
///                       set is CORA's reachable-set blue.
/// The default is "CORA:default"; any other name is an error.
void use_cora_colors(const std::string &scheme);

/// The set S projected onto the two dimensions `dims` (0-based), into the figure `fig`.
void plot(Figure &fig, const ContSet &S, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// The time-interval sets of R, as one region.
void plot(Figure &fig, const ReachSet &R, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// The region a specification forbids, shaded up to the limits of the figure: the outside of
/// each halfspace of a safe set, the halfspace of an unsafe set.
void plot(Figure &fig, const Specification &spec, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// Trajectories as `LinearSys::simulate` returns them, x[k] the points (n, N) at time k: a line
/// per trajectory.
void plot(Figure &fig, const std::vector<Tensor> &x, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// Points as columns (n, N), as `rand_point` returns them.
void plot(Figure &fig, const Tensor &points, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

// The same into the current figure().
inline void plot(const ContSet &S, const std::vector<int64_t> &dims = {0, 1},
                 const PlotOptions &options = {}) {
    plot(figure(), S, dims, options);
}
inline void plot(const ReachSet &R, const std::vector<int64_t> &dims = {0, 1},
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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
