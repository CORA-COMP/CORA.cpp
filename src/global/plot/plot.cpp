// plot - draws sets, reachable sets, specifications and trajectories into a figure
//
// Syntax:   plot(fig, S, dims, options);   plot(fig, R, dims, options);   plot(fig, spec, dims)
//           plot_over_time(fig, R, dim, options);   plot_rand_point(fig, S, dims, N, rng)
// Inputs:   dims - the one, two or three dimensions to show (0-based); options - see PlotOptions
// Outputs:  -, the figure has new layers
// See also: cora/global/plot/figure.h, contSet/zonotope/vertices, contSet/zonotope/project

#include "cora/global/plot/plot.h"
#include "cora/global/error.h"
#include "cora/global/plot/geometry.h"

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
    if (options.color || options.face_color || aux_scheme() == Scheme::standard) {
        alpha = options.face_alpha.value_or(0.2);
        const Color base = options.color        ? *options.color
                           : options.face_color ? *options.face_color
                                                : fig.next_color();
        edge = options.color.value_or(options.face_color ? cora_color("CORA:simulations") : base);
        face = options.face_color.value_or(base);
    }
    return {options.edge_color.value_or(edge), face, alpha};
}

// The color of a reachable set: from the options, else CORA's blue shaded by the scheme, else the
// next color of the figure.
Color aux_reach_color(Figure &fig, const PlotOptions &options) {
    if (options.face_color) return *options.face_color;
    if (options.color) return *options.color;
    if (aux_scheme() == Scheme::cont_dynamics)
        return cora_color("CORA:reachSet", options.num_colors, options.cidx);
    return fig.next_color();
}

// Arguments ---------------------------------------------------------------------------------------

// The figure must have the dimension (2 or 3) that the plot needs; it takes it if it is empty.
void aux_check_figure(Figure &fig, int dimension) {
    if (fig.dimension != dimension && !fig.layers().empty())
        CORA_FAIL(Error::wrong_value, "plot: this figure is " + std::to_string(fig.dimension) +
                                          "-dimensional, but this plot needs " +
                                          std::to_string(dimension) + "; use a figure of its own");
    fig.dimension = dimension;
}

// The number of dimensions must be in fewest..most; the axes are named after them unless they
// already are. The figure becomes three-dimensional for three dimensions.
void aux_check_dims(Figure &fig, const std::vector<int64_t> &dims, std::size_t most = 3,
                    std::size_t fewest = 1) {
    if (dims.size() < fewest || dims.size() > most)
        CORA_FAIL(Error::wrong_value, "plot: dims must name " + std::to_string(fewest) +
                                          (fewest == most ? "" : " to " + std::to_string(most)) +
                                          " dimensions, e.g. {0, 1}, but " +
                                          std::to_string(dims.size()) + " were given");
    aux_check_figure(fig, dims.size() == 3 ? 3 : 2);
    std::string *labels[3] = {&fig.xlabel, &fig.ylabel, &fig.zlabel};
    for (std::size_t i = 0; i < dims.size(); ++i)
        if (labels[i]->empty()) *labels[i] = "x[" + std::to_string(dims[i]) + "]";
}

// The axes of a plot over time: the time and the dimension.
void aux_time_axes(Figure &fig, int64_t dim) {
    aux_check_figure(fig, 2);
    if (fig.xlabel.empty()) fig.xlabel = "t";
    if (fig.ylabel.empty()) fig.ylabel = "x[" + std::to_string(dim) + "]";
}

// The vertices of S projected onto dims, per batch member: a bar for one dimension, as CORA draws
// the interval on the x axis.
std::vector<Polygon> aux_polygons(const ContSet &S, const std::vector<int64_t> &dims) {
    if (dims.size() == 2) return S.project(dims)->vertices();
    const Interval box = S.project(dims)->interval();
    const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
    std::vector<Polygon> bars;
    for (std::size_t b = 0; b < lo.size(); ++b)
        bars.push_back(lo[b] == hi[b] ? Polygon{{lo[b], 0}} : Polygon{{lo[b], 0}, {hi[b], 0}});
    return bars;
}

// The faces of the polytopes of S projected onto three dimensions, per batch member.
std::vector<std::vector<Point3>> aux_faces(const ContSet &S, const std::vector<int64_t> &dims) {
    std::vector<std::vector<Point3>> faces;
    for (const std::vector<Point3> &corners : S.project(dims)->vertices_3d())
        for (std::vector<Point3> &face : hull_faces(corners)) faces.push_back(std::move(face));
    return faces;
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
                                          std::to_string(dims[1]) +
                                          " do not exist in the halfspace");
    return {a[static_cast<std::size_t>(dims[0])], a[static_cast<std::size_t>(dims[1])]};
}

// Reachable sets ----------------------------------------------------------------------------------

// Which sets of R are drawn: "ti" or "tp", and every step-th of them. The time-interval sets are
// replaced by the time-point sets if there are none, with a warning, as CORA does.
const std::vector<Zonotope> &aux_sets(const ReachSet &R, const PlotOptions &options) {
    if (options.set != "ti" && options.set != "tp")
        CORA_FAIL(Error::wrong_value, "plot: set must be \"ti\" (the time-interval sets) or "
                                      "\"tp\" (the time-point sets), but is \"" +
                                          options.set + "\"");
    if (options.step < 1)
        CORA_FAIL(Error::wrong_value,
                  "plot: step must be at least 1, but is " + std::to_string(options.step));
    if (options.set == "ti" && R.time_interval.empty()) {
        warn("CORA:plot",
             "No time-interval reachable set. Time-point reachable set plotted instead.");
        return R.time_point;
    }
    return options.set == "ti" ? R.time_interval : R.time_point;
}

// The range of dimension dim of each set, as (lower, upper) per set and batch member.
std::vector<Point> aux_ranges(const std::vector<Zonotope> &sets, int64_t dim) {
    std::vector<Point> ranges;
    for (const Zonotope &Z : sets) {
        const Interval box = Z.project({dim})->interval();
        const std::vector<double> lo = box.inf().data(), hi = box.sup().data();
        for (std::size_t b = 0; b < lo.size(); ++b) ranges.push_back({lo[b], hi[b]});
    }
    return ranges;
}

// The seconds of a ReachSet are known only if the algorithm recorded its step.
double aux_time_step(const ReachSet &R) {
    if (!(R.time_step > 0))
        CORA_FAIL(Error::wrong_value, "plot_over_time: the reachable set has no time_step; the "
                                      "algorithms of reach set it, or set R.time_step");
    return R.time_step;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Colors ------------------------------------------------------------------------------------------

void use_cora_colors(const std::string &scheme) {
    if (scheme == "CORA:default")
        aux_scheme() = Scheme::standard;
    else if (scheme == "CORA:contDynamics")
        aux_scheme() = Scheme::cont_dynamics;
    else
        CORA_FAIL(Error::wrong_value, "use_cora_colors: unknown color scheme '" + scheme +
                                          "'; use \"CORA:default\" or \"CORA:contDynamics\"");
}

// Sets and Reachable Sets -------------------------------------------------------------------------

void plot(Figure &fig, const ContSet &S, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims);
    const SetColors colors = aux_set_colors(fig, options);
    const std::optional<Color> fill =
        options.filled ? std::optional<Color>(colors.face) : std::nullopt;
    if (dims.size() == 3) {
        fig.add_surface(aux_faces(S, dims), colors.edge, fill, options.line_width.value_or(1.0),
                        options.label)
            .face_alpha = colors.alpha;
        return;
    }
    fig.add_polygons(aux_polygons(S, dims), colors.edge, fill, options.line_width.value_or(1.0),
                     options.label, options.unify)
        .face_alpha = colors.alpha;
}

void plot(Figure &fig, const ReachSet &R, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims, 3, 2);
    const std::vector<Zonotope> &sets = aux_sets(R, options);
    // A reachable set is one region: its fill and its outline have the same color.
    const Color color = aux_reach_color(fig, options);
    const Color edge = options.edge_color.value_or(color);
    const std::optional<Color> fill = options.filled ? std::optional<Color>(color) : std::nullopt;
    if (dims.size() == 3) {
        std::vector<std::vector<Point3>> faces;
        for (std::size_t k = 0; k < sets.size(); k += static_cast<std::size_t>(options.step))
            for (std::vector<Point3> &face : aux_faces(sets[k], dims))
                faces.push_back(std::move(face));
        fig.add_surface(std::move(faces), edge, fill, options.line_width.value_or(1.0),
                        options.label)
            .face_alpha = options.face_alpha.value_or(1.0);
        return;
    }
    // The polygons of the sets, and of the time points if they are asked for.
    std::vector<Polygon> polygons, points;
    for (std::size_t k = 0; k < sets.size(); k += static_cast<std::size_t>(options.step)) {
        const std::vector<Polygon> p = aux_polygons(sets[k], dims);
        polygons.insert(polygons.end(), p.begin(), p.end());
    }
    for (std::size_t k = 0; options.time_points && k < R.time_point.size();
         k += static_cast<std::size_t>(options.step)) {
        const std::vector<Polygon> p = aux_polygons(R.time_point[k], dims);
        points.insert(points.end(), p.begin(), p.end());
    }
    // The region lies under the sets: zorder 1, and opaque unless face_alpha says otherwise.
    fig.equal_axes = true;
    Figure::Layer &region =
        fig.add_polygons(std::move(polygons), edge, fill, options.line_width.value_or(1.0),
                         options.label, options.unify);
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

void plot_over_time(Figure &fig, const ReachSet &R, int64_t dim, const PlotOptions &options) {
    aux_time_axes(fig, dim);
    const std::vector<Zonotope> &sets = aux_sets(R, options);
    const double dt = aux_time_step(R);
    const Color color = aux_reach_color(fig, options);
    const Color edge = options.edge_color.value_or(color);
    const bool intervals = options.set == "ti" && !R.time_interval.empty();
    // Over its time interval a set is a box; a time-point set is a vertical line.
    std::vector<Polygon> shapes;
    const std::vector<Point> ranges = aux_ranges(sets, dim);
    const std::size_t batch = sets.empty() ? 1 : ranges.size() / sets.size();
    for (std::size_t i = 0; i < ranges.size();
         i += batch * static_cast<std::size_t>(options.step)) {
        const double t0 = static_cast<double>(i / batch) * dt, t1 = t0 + dt;
        for (std::size_t b = 0; b < batch; ++b) {
            const double lo = ranges[i + b][0], hi = ranges[i + b][1];
            if (intervals)
                shapes.push_back({{t0, lo}, {t1, lo}, {t1, hi}, {t0, hi}});
            else
                shapes.push_back(lo == hi ? Polygon{{t0, lo}} : Polygon{{t0, lo}, {t0, hi}});
        }
    }
    // Boxes are united into one region, drawn like a reachable set; lines are not filled.
    const std::optional<Color> fill =
        options.filled && intervals ? std::optional<Color>(color) : std::nullopt;
    Figure::Layer &layer =
        fig.add_polygons(std::move(shapes), edge, fill, options.line_width.value_or(1.0),
                         options.label, options.unify);
    layer.zorder = 1;
    layer.face_alpha = options.face_alpha.value_or(1.0);
}

void plot_over_time(Figure &fig, const std::vector<Tensor> &x, int64_t dim, double time_step,
                    const PlotOptions &options) {
    aux_time_axes(fig, dim);
    // Points and a step are needed to place the trajectories on the time axis.
    if (x.empty()) CORA_FAIL(Error::wrong_value, "plot_over_time: no trajectory points given");
    if (!(time_step > 0))
        CORA_FAIL(Error::wrong_value, "plot_over_time: time_step must be positive, but is " +
                                          std::to_string(time_step));
    std::vector<std::vector<Point>> at; // the values at each time
    for (const Tensor &xk : x) at.push_back(aux_points(xk, {dim, dim}));
    const Color color = options.color.value_or(cora_color("CORA:simulations"));
    for (std::size_t j = 0; j < at[0].size(); ++j) {
        std::vector<Point> line;
        for (std::size_t k = 0; k < at.size(); ++k)
            line.push_back({static_cast<double>(k) * time_step, at[k][j][0]});
        fig.add_polyline(std::move(line), color, options.line_width.value_or(0.8),
                         j == 0 ? options.label : "");
    }
}

void plot_rand_point(Figure &fig, const ContSet &S, const std::vector<int64_t> &dims, int64_t N,
                     Rng &rng, const PlotOptions &options) {
    plot(fig, S.rand_point(N, rng), dims, options);
}

// Specifications and Data -------------------------------------------------------------------------

void plot(Figure &fig, const Specification &spec, const std::vector<int64_t> &dims,
          const PlotOptions &options) {
    aux_check_dims(fig, dims, 2, 2);
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
                       options.face_color.value_or(cora_color("CORA:unsafeLight")),
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
