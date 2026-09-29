// plot - draws sets, reachable sets, specifications and trajectories into the current figure
//
// Syntax:   plot(S, dims, options);   plot(R, dims, options);   plot(spec, dims, options)
// Inputs:   dims - the two dimensions to show (0-based); options - see PlotOptions
// Outputs:  -, the figure() has a new layer
// See also: plot/figure.h, contSet/zonotope/vertices, contSet/zonotope/project

#include "plot/plot.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Arguments ---------------------------------------------------------------------------------------

// The dimensions must be two; the axes are named after them unless they already are.
void aux_checkDims(const std::vector<int64_t> &dims) {
    if (dims.size() != 2)
        throw std::invalid_argument("plot: dims must name two dimensions, e.g. {0, 1}, but " +
                                    std::to_string(dims.size()) + " were given");
    Figure &fig = figure();
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

// Sets and Reachable Sets -------------------------------------------------------------------------

void plot(const ContSet &S, const std::vector<int64_t> &dims, const PlotOptions &options) {
    aux_checkDims(dims);
    const std::vector<Polygon> polygons = aux_polygons(S, dims);
    figure().addPolygons(polygons, options.color.value_or(figure().nextColor()), options.facecolor,
                         options.lineWidth.value_or(1.5), options.label, options.unify);
}

void plot(const Reach &R, const std::vector<int64_t> &dims, const PlotOptions &options) {
    aux_checkDims(dims);
    std::vector<Polygon> polygons;
    for (const Zonotope &Z : R.timeInt) {
        const std::vector<Polygon> p = aux_polygons(Z, dims);
        polygons.insert(polygons.end(), p.begin(), p.end());
    }
    // A reachable set is one filled region: its fill and its outline have the same color.
    const Color color =
        options.facecolor.value_or(options.color.value_or(CORAcolor("CORA:reachSet")));
    figure().equalAxes = true;
    figure().addPolygons(std::move(polygons), color, color, options.lineWidth.value_or(1.0),
                         options.label, options.unify);
}

// Specifications and Data -------------------------------------------------------------------------

void plot(const Specification &spec, const std::vector<int64_t> &dims, const PlotOptions &options) {
    aux_checkDims(dims);
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
    bool first = true;
    for (const Halfspace &h : spec.halfspaces()) {
        figure().addRegion(
            aux_normal(h, dims), h.b, sign, options.color.value_or(CORAcolor("CORA:unsafe")),
            options.facecolor.value_or(CORAcolor("CORA:unsafeLight")), first ? options.label : "");
        first = false;
    }
}

void plot(const std::vector<Tensor> &x, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_checkDims(dims);
    if (x.empty()) throw std::invalid_argument("plot: no trajectory points given");
    std::vector<std::vector<Point>> at; // the points at each time, as plane points
    for (const Tensor &xk : x) at.push_back(aux_points(xk, dims));
    const Color color = options.color.value_or(CORAcolor("CORA:simulations"));
    for (std::size_t j = 0; j < at[0].size(); ++j) {
        std::vector<Point> line;
        for (const auto &points : at) line.push_back(points[j]);
        figure().addPolyline(std::move(line), color, options.lineWidth.value_or(0.8),
                             j == 0 ? options.label : "");
    }
    figure().addPoints(at[0], color, 2.0, "");
}

void plot(const Tensor &points, const std::vector<int64_t> &dims, const PlotOptions &options) {
    aux_checkDims(dims);
    figure().addPoints(aux_points(points, dims),
                       options.color.value_or(CORAcolor("CORA:simulations")),
                       options.lineWidth.value_or(2.0), options.label);
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
