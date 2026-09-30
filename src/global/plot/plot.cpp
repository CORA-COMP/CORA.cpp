// plot - draws sets, reachable sets, specifications and trajectories into a figure
//
// Syntax:   plot(fig, S, dims, options);   plot(fig, R, dims, options);   plot(fig, spec, dims)
// Inputs:   dims - the two dimensions to show (0-based); options - see PlotOptions
// Outputs:  -, the figure has new layers
// See also: global/plot/figure.h, contSet/zonotope/vertices, contSet/zonotope/project

#include "global/plot/plot.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The color scheme ----------------------------------------------------------------------------

enum class Scheme { Default, ContDynamics };

Scheme &aux_scheme() {
    static Scheme scheme = Scheme::Default;
    return scheme;
}

// The outline, the fill and its opacity of a set: from the options if given; else, with the
// scheme "CORA:contDynamics", an initial set (opaque white); else the next color of the figure
// for both, with a fill of opacity 0.2.
struct SetColors {
    Color edge, face;
    double alpha;
};

SetColors aux_setColors(Figure &fig, const PlotOptions &options) {
    Color edge = CORAcolor("CORA:simulations"), face = CORAcolor("CORA:initialSet");
    double alpha = options.faceAlpha.value_or(1.0);
    if (options.color || options.facecolor || aux_scheme() == Scheme::Default) {
        alpha = options.faceAlpha.value_or(0.2);
        const Color base = options.color ? *options.color
                           : options.facecolor ? *options.facecolor
                                               : fig.nextColor();
        edge = options.color.value_or(options.facecolor ? CORAcolor("CORA:simulations") : base);
        face = options.facecolor.value_or(base);
    }
    return {edge, face, alpha};
}

// The color of a reachable set: from the options, else CORA's blue shaded by the scheme, else the
// next color of the figure.
Color aux_reachColor(Figure &fig, const PlotOptions &options) {
    if (options.facecolor) return *options.facecolor;
    if (options.color) return *options.color;
    if (aux_scheme() == Scheme::ContDynamics)
        return CORAcolor("CORA:reachSet", options.numColors, options.cidx);
    return fig.nextColor();
}

// Arguments ---------------------------------------------------------------------------------------

// The dimensions must be two; the axes are named after them unless they already are.
void aux_checkDims(Figure &fig, const std::vector<int64_t> &dims) {
    if (dims.size() != 2)
        throw std::invalid_argument("plot: dims must name two dimensions, e.g. {0, 1}, but " +
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
        throw std::invalid_argument("plot: points must be a matrix (n, N) without batch "
                                    "dimensions, but have " +
                                    std::to_string(shape.size()) + " dimensions");
    if (dims[0] >= shape[0] || dims[1] >= shape[0] || dims[0] < 0 || dims[1] < 0)
        throw std::invalid_argument("plot: the dimensions " + std::to_string(dims[0]) + " and " +
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
        throw std::invalid_argument("plot: the dimensions " + std::to_string(dims[0]) + " and " +
                                    std::to_string(dims[1]) + " do not exist in the halfspace");
    return {a[static_cast<std::size_t>(dims[0])], a[static_cast<std::size_t>(dims[1])]};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Colors ------------------------------------------------------------------------------------------

void useCORAcolors(const std::string &scheme) {
    if (scheme == "CORA:default") aux_scheme() = Scheme::Default;
    else if (scheme == "CORA:contDynamics") aux_scheme() = Scheme::ContDynamics;
    else
        throw std::invalid_argument("useCORAcolors: unknown color scheme '" + scheme +
                                    "'; use \"CORA:default\" or \"CORA:contDynamics\"");
}

// Sets and Reachable Sets -------------------------------------------------------------------------

void plot(Figure &fig, const ContSet &S, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_checkDims(fig, dims);
    const std::vector<Polygon> polygons = aux_polygons(S, dims);
    const SetColors colors = aux_setColors(fig, options);
    fig.addPolygons(polygons, colors.edge,
                    options.filled ? std::optional<Color>(colors.face) : std::nullopt,
                    options.lineWidth.value_or(1.0), options.label, options.unify)
        .faceAlpha = colors.alpha;
}

void plot(Figure &fig, const Reach &R, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_checkDims(fig, dims);
    if (options.step < 1)
        throw std::invalid_argument("plot: step must be at least 1, but is " +
                                    std::to_string(options.step));
    // A reachable set is one region: its fill and its outline have the same color.
    const Color color = aux_reachColor(fig, options);
    std::vector<Polygon> polygons, points;
    for (std::size_t k = 0; k < R.timeInt.size(); k += options.step) {
        const std::vector<Polygon> p = aux_polygons(R.timeInt[k], dims);
        polygons.insert(polygons.end(), p.begin(), p.end());
    }
    for (std::size_t k = 0; options.timePoints && k < R.timePoint.size(); k += options.step) {
        const std::vector<Polygon> p = aux_polygons(R.timePoint[k], dims);
        points.insert(points.end(), p.begin(), p.end());
    }
    fig.equalAxes = true;
    Figure::Layer &region = fig.addPolygons(
        std::move(polygons), color, options.filled ? std::optional<Color>(color) : std::nullopt,
        options.lineWidth.value_or(1.0), options.label, options.unify);
    region.zorder = 1;
    region.faceAlpha = options.faceAlpha.value_or(1.0);
    // The time points lie between the reachable set and the sets on top of it.
    if (options.timePoints) {
        Figure::Layer &layer = fig.addPolygons(std::move(points), CORAcolor("CORA:reachSet:dark"),
                                               std::nullopt, 0.5, "", false);
        layer.zorder = 2;
        layer.dashed = true;
    }
}

// Specifications and Data -------------------------------------------------------------------------

void plot(Figure &fig, const Specification &spec, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_checkDims(fig, dims);
    // A halfspace is a'x <= b; a safe set forbids a'x > b, an unsafe set forbids a'x <= b.
    double sign = 0;
    switch (spec.type()) {
    case SpecType::SafeSet:
        sign = 1;
        break;
    case SpecType::UnsafeSet:
        sign = -1;
        break;
    default:
        throw std::invalid_argument("plot: unknown specification type; expected SafeSet or "
                                    "UnsafeSet");
    }
    // Only the first region carries the label, so the legend names the specification once.
    bool first = true;
    for (const Halfspace &h : spec.halfspaces()) {
        fig.addRegion(aux_normal(h, dims), h.b, sign,
                      options.color.value_or(CORAcolor("CORA:unsafe")),
                      options.facecolor.value_or(CORAcolor("CORA:unsafeLight")),
                      first ? options.label : "");
        first = false;
    }
}

void plot(Figure &fig, const std::vector<Tensor> &x, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_checkDims(fig, dims);
    if (x.empty()) throw std::invalid_argument("plot: no trajectory points given");
    std::vector<std::vector<Point>> at; // the points at each time, as plane points
    for (const Tensor &xk : x) at.push_back(aux_points(xk, dims));
    const Color color = options.color.value_or(CORAcolor("CORA:simulations"));
    for (std::size_t j = 0; j < at[0].size(); ++j) {
        std::vector<Point> line;
        for (const auto &points : at) line.push_back(points[j]);
        fig.addPolyline(std::move(line), color, options.lineWidth.value_or(0.8),
                        j == 0 ? options.label : "");
    }
}

void plot(Figure &fig, const Tensor &points, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_checkDims(fig, dims);
    fig.addPoints(aux_points(points, dims), options.color.value_or(CORAcolor("CORA:simulations")),
                  options.lineWidth.value_or(2.0), options.label);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
