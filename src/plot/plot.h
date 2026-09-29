// plot - draws sets, reachable sets, specifications and trajectories into the current figure
//
// As CORA's plot: the set is projected onto two dimensions, its vertices are computed, and the
// vertices are drawn with the plot options. Reachable sets are drawn as the union of their sets.
//
// Syntax:   plot(X0, {0, 1});   plot(R, {0, 1}, {.label = "reachable set"});
//           plot(spec, {0, 1});   plot(x, {0, 1});   figure().save("figure.svg");
// See also: plot/figure.h, plot/color.h, contSet/contSet.h

#pragma once

#include "contDynamics/linearSys/linearSys.h"
#include "plot/figure.h"
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

    /// The color of the outline (of a set), of the lines (of trajectories) or of the fill (of a
    /// reachable set); a CORA color converts implicitly, e.g. `{.color = "CORA:red"}`.
    std::optional<Color> color = std::nullopt;

    /// The fill of a set; a set is an outline only without it.
    std::optional<Color> facecolor = std::nullopt;

    /// The width of lines and outlines in pixels.
    std::optional<double> lineWidth = std::nullopt;

    /// Whether the sets of a reachable set are drawn as their union.
    bool unify = true;
};

/// The set S projected onto the two dimensions `dims` (0-based): its outline, or its fill too.
void plot(const ContSet &S, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// The time-interval sets of R, as one region in CORA's reachable-set blue.
void plot(const Reach &R, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// The region a specification forbids, shaded up to the limits of the figure: the outside of
/// each halfspace of a safe set, the halfspace of an unsafe set.
void plot(const Specification &spec, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// Trajectories as `LinearSys::simulate` returns them, x[k] the points (n, N) at time k: a line
/// per trajectory, and a dot where each starts.
void plot(const std::vector<Tensor> &x, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

/// Points as columns (n, N), as `randPoint` returns them.
void plot(const Tensor &points, const std::vector<int64_t> &dims = {0, 1},
          const PlotOptions &options = {});

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
