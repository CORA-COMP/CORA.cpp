// figure - the layers of one pair of axes and the limits they give
//
// Syntax:   figure().add_polygons(...);   Figure::Limits l = figure().limits();
// Inputs:   the layers, in the order they are drawn: regions first, then sets, lines, points
// Outputs:  the layers, the limits and the view of the axes
// See also: cora/global/plot/plot.h, src/global/plot/svg.cpp, src/global/plot/tikz.cpp

#include "cora/global/plot/figure.h"
#include "cora/global/error.h"
#include "cora/global/plot/geometry.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The range lo..hi widened by 5% on both sides.
void aux_pad(double &lo, double &hi) {
    const double w = hi - lo > numerics::kAxisRange ? hi - lo : std::max(1.0, std::abs(lo) * 0.1);
    lo -= 0.05 * w;
    hi += 0.05 * w;
}

// The limit that the user gave, if any.
void aux_override(const std::optional<std::array<double, 2>> &lim, double &lo, double &hi) {
    if (lim) {
        lo = (*lim)[0];
        hi = (*lim)[1];
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Layers ------------------------------------------------------------------------------------------

Figure::Layer &Figure::add_polygons(std::vector<Polygon> polygons, Color edge,
                                    std::optional<Color> face, double line_width,
                                    const std::string &label, bool unify) {
    // A union has no inner edges, and its outline is the outline of the whole.
    if (unify && polygons.size() > 1) {
        std::vector<Polygon> degenerate; // points and segments have no area to unite
        std::vector<Polygon> areas;
        for (Polygon &p : polygons) (p.size() < 3 ? degenerate : areas).push_back(std::move(p));
        polygons = union_polygons(areas);
        polygons.insert(polygons.end(), degenerate.begin(), degenerate.end());
    }
    Layer l{Kind::polygons, std::move(polygons), {}, edge, face.value_or(edge)};
    l.filled = face.has_value();
    l.unify = unify;
    l.line_width = line_width;
    l.label = label;
    layers_.push_back(std::move(l));
    return layers_.back();
}

Figure::Layer &Figure::add_surface(std::vector<std::vector<Point3>> faces, Color edge,
                                   std::optional<Color> face, double line_width,
                                   const std::string &label) {
    if (dimension != 3 && !layers_.empty())
        CORA_FAIL(Error::wrong_value, "Figure: a surface needs a three-dimensional figure, but "
                                      "this one has two-dimensional layers");
    // The figure is three-dimensional from now on; a later 2-D plot is refused.
    dimension = 3;
    Layer l{Kind::surface, {}, std::move(faces), edge, face.value_or(edge)};
    l.filled = face.has_value();
    l.line_width = line_width;
    l.label = label;
    layers_.push_back(std::move(l));
    return layers_.back();
}

// Lines Points and Regions ------------------------------------------------------------------------

// A polyline is a layer with one polygon that is not closed.
Figure::Layer &Figure::add_polyline(std::vector<Point> points, Color color, double line_width,
                                    const std::string &label) {
    Layer l{Kind::polyline, {Polygon(points.begin(), points.end())}, {}, color, color};
    l.line_width = line_width;
    l.label = label;
    l.zorder = 4;
    layers_.push_back(std::move(l));
    return layers_.back();
}

// Points are a layer with one polygon whose vertices are the dots.
Figure::Layer &Figure::add_points(std::vector<Point> points, Color color, double radius,
                                  const std::string &label) {
    Layer l{Kind::points, {Polygon(points.begin(), points.end())}, {}, color, color};
    l.radius = radius;
    l.label = label;
    l.zorder = 4;
    layers_.push_back(std::move(l));
    return layers_.back();
}

// A region keeps its halfspace; it is cut to the limits when the figure is written.
Figure::Layer &Figure::add_region(Point a, double b, double sign, Color edge, Color face,
                                  const std::string &label) {
    Layer l{Kind::region, {}, {}, edge, face};
    l.a = a;
    l.b = b;
    l.sign = sign;
    l.label = label;
    l.zorder = 0;
    layers_.push_back(std::move(l));
    return layers_.back();
}

// The palette of CORA repeats after its seventh color.
Color Figure::next_color() { return cora_color("CORA:next", 1, ++colors_given_); }

std::vector<Figure::Layer> Figure::take_layers() { return std::exchange(layers_, {}); }

// The part of the polygon where sign * (a.x - b) >= 0, cut edge by edge.
Polygon Figure::clip_region(const Polygon &poly, Point a, double b, double sign) {
    auto value = [&](const Point &p) { return sign * (a[0] * p[0] + a[1] * p[1] - b); };
    Polygon out;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const Point &p = poly[i], &q = poly[(i + 1) % poly.size()];
        const double vp = value(p), vq = value(q);
        if (vp >= 0) out.push_back(p);
        if ((vp >= 0) != (vq >= 0)) {
            const double t = vp / (vp - vq);
            out.push_back({p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])});
        }
    }
    return out;
}

void Figure::clear() { *this = Figure(); }

Figure &figure() {
    static Figure current;
    return current;
}

// Limits and View ---------------------------------------------------------------------------------

// The data limits with a margin; with equal axes, widened so that one unit is as long on both.
Figure::Limits Figure::limits(double pw, double ph) const {
    Limits l{1e300, -1e300, 1e300, -1e300, 1e300, -1e300};
    auto extend = [&](double x, double y, double z) {
        l.x0 = std::min(l.x0, x);
        l.x1 = std::max(l.x1, x);
        l.y0 = std::min(l.y0, y);
        l.y1 = std::max(l.y1, y);
        l.z0 = std::min(l.z0, z);
        l.z1 = std::max(l.z1, z);
    };
    for (const Layer &layer : layers_) {
        for (const Polygon &poly : layer.polygons)
            for (const Point &p : poly) extend(p[0], p[1], 0);
        for (const std::vector<Point3> &face : layer.faces)
            for (const Point3 &p : face) extend(p[0], p[1], p[2]);
    }
    if (l.x0 > l.x1) l = {-1, 1, -1, 1, -1, 1}; // nothing drawn yet
    aux_pad(l.x0, l.x1);
    aux_pad(l.y0, l.y1);
    aux_pad(l.z0, l.z1);
    if (equal_axes && dimension == 2 && pw > 0 && ph > 0) {
        const double per_pixel_x = (l.x1 - l.x0) / pw, per_pixel_y = (l.y1 - l.y0) / ph;
        const double s = std::max(per_pixel_x, per_pixel_y);
        const double cx = (l.x0 + l.x1) / 2, cy = (l.y0 + l.y1) / 2;
        l = {cx - s * pw / 2, cx + s * pw / 2, cy - s * ph / 2, cy + s * ph / 2, l.z0, l.z1};
    }
    aux_override(xlim, l.x0, l.x1);
    aux_override(ylim, l.y0, l.y1);
    aux_override(zlim, l.z0, l.z1);
    return l;
}

// The orthographic view from the direction of MATLAB's view(azimuth, elevation): the azimuth is
// turned from the -y axis, counter-clockwise seen from above.
Point Figure::view(const Point3 &p, double *depth) const {
    const double pi = std::acos(-1.0);
    const double a = (azimuth - 90) * pi / 180, e = elevation * pi / 180;
    if (depth)
        *depth = (p[0] * std::cos(a) + p[1] * std::sin(a)) * std::cos(e) + p[2] * std::sin(e);
    return {-p[0] * std::sin(a) + p[1] * std::cos(a),
            -(p[0] * std::cos(a) + p[1] * std::sin(a)) * std::sin(e) + p[2] * std::cos(e)};
}

// The Grid
// -----------------------------------------------------------------------------------------

FigureGrid::FigureGrid(int rows, int cols) : rows_(rows), cols_(cols) {
    if (rows < 1 || cols < 1)
        CORA_FAIL(Error::wrong_value, "FigureGrid: needs at least one row and one column, got " +
                                          std::to_string(rows) + " x " + std::to_string(cols));
    cells_.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
}

const Figure &FigureGrid::at(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        CORA_FAIL(Error::wrong_value, "FigureGrid: the cell (" + std::to_string(row) + ", " +
                                          std::to_string(col) + ") does not exist in " +
                                          std::to_string(rows_) + " x " + std::to_string(cols_));
    return cells_[static_cast<std::size_t>(row * cols_ + col)];
}

Figure &FigureGrid::at(int row, int col) {
    return const_cast<Figure &>(static_cast<const FigureGrid &>(*this).at(row, col));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
