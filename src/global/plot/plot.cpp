// plot - draws sets, reachable sets, specifications and trajectories into a figure
//
// Syntax:   plot(fig, S, dims, options);   plot(fig, R, dims, options);   plot(fig, spec, dims)
// Inputs:   dims - the two dimensions to show (0-based); options - see PlotOptions
// Outputs:  -, the figure has new layers
// See also: cora/global/plot/figure.h, contSet/zonotope/vertices, contSet/zonotope/project

#include "cora/global/plot/plot.h"
#include "cora/global/error.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The color scheme ----------------------------------------------------------------------------

enum class Scheme { standard, cont_dynamics };

Scheme &aux_scheme() {
    static Scheme scheme = Scheme::standard;
    return scheme;
}

// The outline, the fill and its opacity of a set: from the options if given; else, with the
// scheme "CORA:contDynamics", an initial set (opaque white); else the next color of the figure
// for both, with a fill of opacity 0.2.
struct SetColors {
    Color edge, face;
    double alpha;
};

SetColors aux_set_colors(Figure &fig, const PlotOptions &options) {
    Color edge = cora_color("CORA:simulations"), face = cora_color("CORA:initialSet");
    double alpha = options.face_alpha.value_or(1.0);
    if (options.color || options.facecolor || aux_scheme() == Scheme::standard) {
        alpha = options.face_alpha.value_or(0.2);
        const Color base = options.color ? *options.color
                           : options.facecolor ? *options.facecolor
                                               : fig.next_color();
        edge = options.color.value_or(options.facecolor ? cora_color("CORA:simulations") : base);
        face = options.facecolor.value_or(base);
    }
    return {edge, face, alpha};
}

// The color of a reachable set: from the options, else CORA's blue shaded by the scheme, else the
// next color of the figure.
Color aux_reach_color(Figure &fig, const PlotOptions &options) {
    if (options.facecolor) return *options.facecolor;
    if (options.color) return *options.color;
    if (aux_scheme() == Scheme::cont_dynamics)
        return cora_color("CORA:reachSet", options.num_colors, options.cidx);
    return fig.next_color();
}

// Arguments ---------------------------------------------------------------------------------------

// The dimensions must be two; the axes are named after them unless they already are.
void aux_check_dims(Figure &fig, const std::vector<int64_t> &dims) {
    if (dims.size() != 2)
        CORA_FAIL(Error::wrong_value, "plot: dims must name two dimensions, e.g. {0, 1}, but " +
                                    std::to_string(dims.size()) + " were given");
    if (fig.xlabel.empty()) fig.xlabel = "x[" + std::to_string(dims[0]) + "]";
    if (fig.ylabel.empty()) fig.ylabel = "x[" + std::to_string(dims[1]) + "]";
}

// The vertices of S projected onto dims, per batch member.
std::vector<Polygon> aux_polygons(const ContSet &S, const std::vector<int64_t> &dims) {
    return S.project(dims)->vertices();
}

// The values of a point set (n, N) or of x[k] over time, as the plane points of dims.
std::vector<Point> aux_points(const Tensor &points, const std::vector<int64_t> &dims) {
    const std::vector<int64_t> shape = points.shape();
    if (shape.size() != 2)
        CORA_FAIL(Error::wrong_value, "plot: points must be a matrix (n, N) without batch "
                                    "dimensions, but have " +
                                    std::to_string(shape.size()) + " dimensions");
    if (dims[0] >= shape[0] || dims[1] >= shape[0] || dims[0] < 0 || dims[1] < 0)
        CORA_FAIL(Error::wrong_value, "plot: the dimensions " + std::to_string(dims[0]) + " and " +
                                    std::to_string(dims[1]) + " do not exist in " +
                                    std::to_string(shape[0]) + "-dimensional points");
    const std::vector<double> v = points.data();
    const int64_t N = shape[1];
    std::vector<Point> out;
    for (int64_t j = 0; j < N; ++j)
        out.push_back({v[static_cast<std::size_t>(dims[0] * N + j)],
                       v[static_cast<std::size_t>(dims[1] * N + j)]});
    return out;
}

// The halfspace's normal vector on the dimensions dims.
Point aux_normal(const Halfspace &h, const std::vector<int64_t> &dims) {
    const std::vector<double> a = h.a.data();
    if (dims[0] >= static_cast<int64_t>(a.size()) || dims[1] >= static_cast<int64_t>(a.size()) ||
        dims[0] < 0 || dims[1] < 0)
        CORA_FAIL(Error::wrong_value, "plot: the dimensions " + std::to_string(dims[0]) + " and " +
                                    std::to_string(dims[1]) + " do not exist in the halfspace");
    return {a[static_cast<std::size_t>(dims[0])], a[static_cast<std::size_t>(dims[1])]};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Colors ------------------------------------------------------------------------------------------

void use_cora_colors(const std::string &scheme) {
    if (scheme == "CORA:default") aux_scheme() = Scheme::standard;
    else if (scheme == "CORA:contDynamics") aux_scheme() = Scheme::cont_dynamics;
    else
        CORA_FAIL(Error::wrong_value, "use_cora_colors: unknown color scheme '" + scheme +
                                    "'; use \"CORA:default\" or \"CORA:contDynamics\"");
}

// Sets and Reachable Sets -------------------------------------------------------------------------

void plot(Figure &fig, const ContSet &S, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims);
    const std::vector<Polygon> polygons = aux_polygons(S, dims);
    const SetColors colors = aux_set_colors(fig, options);
    fig.add_polygons(polygons, colors.edge,
                    options.filled ? std::optional<Color>(colors.face) : std::nullopt,
                    options.line_width.value_or(1.0), options.label, options.unify)
        .face_alpha = colors.alpha;
}

void plot(Figure &fig, const ReachSet &R, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims);
    if (options.step < 1)
        CORA_FAIL(Error::wrong_value, "plot: step must be at least 1, but is " +
                                    std::to_string(options.step));
    // A reachable set is one region: its fill and its outline have the same color.
    const Color color = aux_reach_color(fig, options);
    std::vector<Polygon> polygons, points;
    for (std::size_t k = 0; k < R.time_interval.size(); k += options.step) {
        const std::vector<Polygon> p = aux_polygons(R.time_interval[k], dims);
        polygons.insert(polygons.end(), p.begin(), p.end());
    }
    for (std::size_t k = 0; options.time_points && k < R.time_point.size(); k += options.step) {
        const std::vector<Polygon> p = aux_polygons(R.time_point[k], dims);
        points.insert(points.end(), p.begin(), p.end());
    }
    fig.equal_axes = true;
    Figure::Layer &region = fig.add_polygons(
        std::move(polygons), color, options.filled ? std::optional<Color>(color) : std::nullopt,
        options.line_width.value_or(1.0), options.label, options.unify);
    region.zorder = 1;
    region.face_alpha = options.face_alpha.value_or(1.0);
    // The time points lie between the reachable set and the sets on top of it.
    if (options.time_points) {
        Figure::Layer &layer = fig.add_polygons(std::move(points), cora_color("CORA:reachSet:dark"),
                                               std::nullopt, 0.5, "", false);
        layer.zorder = 2;
        layer.dashed = true;
    }
}

// Specifications and Data -------------------------------------------------------------------------

void plot(Figure &fig, const Specification &spec, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims);
    // A halfspace is a'x <= b; a safe set forbids a'x > b, an unsafe set forbids a'x <= b.
    double sign = 0;
    switch (spec.type()) {
    case SpecType::safe_set:
        sign = 1;
        break;
    case SpecType::unsafe_set:
        sign = -1;
        break;
    default:
        CORA_FAIL(Error::wrong_value, "plot: unknown specification type; expected SafeSet or "
                                    "UnsafeSet");
    }
    // Only the first region carries the label, so the legend names the specification once.
    bool first = true;
    for (const Halfspace &h : spec.halfspaces()) {
        fig.add_region(aux_normal(h, dims), h.b, sign,
                      options.color.value_or(cora_color("CORA:unsafe")),
                      options.facecolor.value_or(cora_color("CORA:unsafeLight")),
                      first ? options.label : "");
        first = false;
    }
}

void plot(Figure &fig, const std::vector<Tensor> &x, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims);
    if (x.empty()) CORA_FAIL(Error::wrong_value, "plot: no trajectory points given");
    std::vector<std::vector<Point>> at; // the points at each time, as plane points
    for (const Tensor &xk : x) at.push_back(aux_points(xk, dims));
    const Color color = options.color.value_or(cora_color("CORA:simulations"));
    for (std::size_t j = 0; j < at[0].size(); ++j) {
        std::vector<Point> line;
        for (const auto &points : at) line.push_back(points[j]);
        fig.add_polyline(std::move(line), color, options.line_width.value_or(0.8),
                        j == 0 ? options.label : "");
    }
}

void plot(Figure &fig, const Tensor &points, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims);
    fig.add_points(aux_points(points, dims), options.color.value_or(cora_color("CORA:simulations")),
                  options.line_width.value_or(2.0), options.label);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
