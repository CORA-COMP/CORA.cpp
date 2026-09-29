// figure - a two-dimensional figure that is written as SVG
//
// Syntax:   figure().addPolygons(...);   figure().save("figure.svg");
// Inputs:   the layers, in the order they are drawn: regions first, then sets, lines, points
// Outputs:  the SVG text, or the file
// See also: plot/plot.h

#include "plot/figure.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Limits ------------------------------------------------------------------------------------------

struct Limits {
    double x0, x1, y0, y1;
};

// The data limits with a margin; with equal axes, widened so that one unit is as long on both.
Limits aux_limits(const std::vector<Polygon> &all, bool equal, double pw, double ph) {
    Limits l{1e300, -1e300, 1e300, -1e300};
    for (const Polygon &poly : all)
        for (const Point &p : poly) {
            l.x0 = std::min(l.x0, p[0]);
            l.x1 = std::max(l.x1, p[0]);
            l.y0 = std::min(l.y0, p[1]);
            l.y1 = std::max(l.y1, p[1]);
        }
    if (l.x0 > l.x1) l = {-1, 1, -1, 1}; // nothing drawn yet
    auto pad = [](double &lo, double &hi) {
        const double w = hi - lo > 1e-12 ? hi - lo : std::max(1e-6, std::abs(lo) * 0.1);
        lo -= 0.05 * w;
        hi += 0.05 * w;
    };
    pad(l.x0, l.x1);
    pad(l.y0, l.y1);
    if (equal) {
        const double perPixelX = (l.x1 - l.x0) / pw, perPixelY = (l.y1 - l.y0) / ph;
        const double s = std::max(perPixelX, perPixelY);
        const double cx = (l.x0 + l.x1) / 2, cy = (l.y0 + l.y1) / 2;
        l = {cx - s * pw / 2, cx + s * pw / 2, cy - s * ph / 2, cy + s * ph / 2};
    }
    return l;
}

// Text and Ticks ----------------------------------------------------------------------------------

// A number with up to four significant digits and no trailing zeros.
std::string aux_num(double v) {
    char out[32];
    std::snprintf(out, sizeof out, "%.4g", std::abs(v) < 1e-12 ? 0.0 : v);
    return out;
}

std::string aux_escape(const std::string &s) {
    std::string out;
    for (char c : s) {
        if (c == '&')
            out += "&amp;";
        else if (c == '<')
            out += "&lt;";
        else if (c == '>')
            out += "&gt;";
        else
            out += c;
    }
    return out;
}

// About six ticks at 1, 2 or 5 times a power of ten.
std::vector<double> aux_ticks(double lo, double hi) {
    const double raw = (hi - lo) / 6, mag = std::pow(10.0, std::floor(std::log10(raw)));
    const double step = (raw / mag < 1.5   ? 1
                         : raw / mag < 3.5 ? 2
                         : raw / mag < 7.5 ? 5
                                           : 10) *
                        mag;
    std::vector<double> ticks;
    for (double t = std::ceil(lo / step) * step; t <= hi + 1e-9 * step; t += step)
        ticks.push_back(t);
    return ticks;
}

// Regions -----------------------------------------------------------------------------------------

// The part of the polygon where sign * (a.x - b) >= 0, cut edge by edge.
Polygon aux_clip(const Polygon &poly, const Point &a, double b, double sign) {
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

} // namespace

// ===========================================  MAIN  =========================================== //

// Layers ------------------------------------------------------------------------------------------

void Figure::addPolygons(std::vector<Polygon> polygons, Color edge, std::optional<Color> face,
                         double lineWidth, const std::string &label, bool unify) {
    Layer l{Kind::Polygons, std::move(polygons), edge, face.value_or(edge)};
    l.filled = face.has_value();
    l.unify = unify;
    l.lineWidth = lineWidth;
    l.label = label;
    layers_.push_back(std::move(l));
}

// A polyline is a layer with one polygon that is not closed.
void Figure::addPolyline(std::vector<Point> points, Color color, double lineWidth,
                         const std::string &label) {
    Layer l{Kind::Polyline, {Polygon(points.begin(), points.end())}, color, color};
    l.lineWidth = lineWidth;
    l.label = label;
    layers_.push_back(std::move(l));
}

// Points are a layer with one polygon whose vertices are the dots.
void Figure::addPoints(std::vector<Point> points, Color color, double radius,
                       const std::string &label) {
    Layer l{Kind::Points, {Polygon(points.begin(), points.end())}, color, color};
    l.radius = radius;
    l.label = label;
    layers_.push_back(std::move(l));
}

// A region keeps its halfspace; it is cut to the limits when the figure is written.
void Figure::addRegion(Point a, double b, double sign, Color edge, Color face,
                       const std::string &label) {
    Layer l{Kind::Region, {}, edge, face};
    l.a = a;
    l.b = b;
    l.sign = sign;
    l.label = label;
    layers_.push_back(std::move(l));
}

void Figure::clear() { *this = Figure(); }

Figure &figure() {
    static Figure current;
    return current;
}

// Writing -----------------------------------------------------------------------------------------

std::string Figure::svg() const {
    const double left = 70, right = 20, top = title.empty() ? 20 : 40, bottom = 55;
    const double pw = width - left - right, ph = height - top - bottom;
    std::vector<Polygon> all; // everything that has an extent, for the limits
    for (const Layer &l : layers_)
        if (l.kind != Kind::Region) all.insert(all.end(), l.polygons.begin(), l.polygons.end());
    const Limits lim = aux_limits(all, equalAxes, pw, ph);
    auto X = [&](double x) { return left + (x - lim.x0) / (lim.x1 - lim.x0) * pw; };
    auto Y = [&](double y) { return top + (lim.y1 - y) / (lim.y1 - lim.y0) * ph; };
    auto path = [&](const Polygon &p, bool close) {
        std::string d;
        for (std::size_t i = 0; i < p.size(); ++i)
            d += (i ? " L" : "M") + aux_num(X(p[i][0])) + " " + aux_num(Y(p[i][1]));
        return d + (close ? " Z" : "");
    };

    std::ostringstream out;
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << " " << height << "\" font-family=\"sans-serif\" "
        << "font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n"
        << "<clipPath id=\"area\"><rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << pw
        << "\" height=\"" << ph << "\"/></clipPath>\n<g clip-path=\"url(#area)\">\n";

    // Regions first, so that the sets lie on top of them.
    const Polygon box = {{lim.x0, lim.y0}, {lim.x1, lim.y0}, {lim.x1, lim.y1}, {lim.x0, lim.y1}};
    for (int pass = 0; pass < 2; ++pass)
        for (const Layer &l : layers_) {
            if ((l.kind == Kind::Region) != (pass == 0)) continue;
            switch (l.kind) {
            case Kind::Region: {
                const Polygon r = aux_clip(box, l.a, l.b, l.sign);
                if (r.size() >= 3)
                    out << "<path d=\"" << path(r, true) << "\" fill=\"" << l.face.hex()
                        << "\" stroke=\"" << l.edge.hex() << "\" stroke-width=\"1.2\"/>\n";
                break;
            }
            case Kind::Polygons: {
                std::string d; // all polygons in one path: its fill is their union
                for (const Polygon &p : l.polygons)
                    if (p.size() >= 2) d += path(p, true) + " ";
                for (const Polygon &p : l.polygons)
                    if (p.size() == 1)
                        out << "<circle cx=\"" << aux_num(X(p[0][0])) << "\" cy=\""
                            << aux_num(Y(p[0][1])) << "\" r=\"1.5\" fill=\"" << l.edge.hex()
                            << "\"/>\n";
                if (d.empty()) break;
                if (l.filled && l.unify) {
                    // The outline under the fill: the fill covers the inner edges and half of
                    // the stroke, leaving the outside of the union's boundary.
                    out << "<path d=\"" << d << "\" fill=\"" << l.face.hex() << "\" stroke=\""
                        << l.edge.hex() << "\" stroke-width=\"" << 2 * l.lineWidth
                        << "\" stroke-linejoin=\"round\" fill-rule=\"nonzero\"/>\n<path d=\"" << d
                        << "\" fill=\"" << l.face.hex() << "\" fill-rule=\"nonzero\"/>\n";
                } else {
                    out << "<path d=\"" << d << "\" fill=\""
                        << (l.filled ? l.face.hex() : std::string("none")) << "\" stroke=\""
                        << l.edge.hex() << "\" stroke-width=\"" << l.lineWidth
                        << "\" stroke-linejoin=\"round\"/>\n";
                }
                break;
            }
            // A line through the points, without fill.
            case Kind::Polyline:
                out << "<path d=\"" << path(l.polygons[0], false) << "\" fill=\"none\" stroke=\""
                    << l.edge.hex() << "\" stroke-width=\"" << l.lineWidth << "\"/>\n";
                break;
            // A dot per point.
            case Kind::Points:
                for (const Point &p : l.polygons[0])
                    out << "<circle cx=\"" << aux_num(X(p[0])) << "\" cy=\"" << aux_num(Y(p[1]))
                        << "\" r=\"" << l.radius << "\" fill=\"" << l.edge.hex() << "\"/>\n";
                break;
            default:
                throw std::logic_error("Figure::svg: unknown layer kind; this is a bug in Figure");
            }
        }
    out << "</g>\n";

    // Axes: the frame, ticks with labels, and the axis titles.
    out << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << pw << "\" height=\"" << ph
        << "\" fill=\"none\" stroke=\"black\"/>\n";
    for (double t : aux_ticks(lim.x0, lim.x1))
        out << "<line x1=\"" << aux_num(X(t)) << "\" y1=\"" << top + ph << "\" x2=\""
            << aux_num(X(t)) << "\" y2=\"" << top + ph + 5 << "\" stroke=\"black\"/>\n<text x=\""
            << aux_num(X(t)) << "\" y=\"" << top + ph + 19 << "\" text-anchor=\"middle\">"
            << aux_num(t) << "</text>\n";
    // The ticks of the y axis, on the left.
    for (double t : aux_ticks(lim.y0, lim.y1))
        out << "<line x1=\"" << left - 5 << "\" y1=\"" << aux_num(Y(t)) << "\" x2=\"" << left
            << "\" y2=\"" << aux_num(Y(t)) << "\" stroke=\"black\"/>\n<text x=\"" << left - 9
            << "\" y=\"" << aux_num(Y(t) + 4) << "\" text-anchor=\"end\">" << aux_num(t)
            << "</text>\n";
    out << "<text x=\"" << left + pw / 2 << "\" y=\"" << height - 12
        << "\" text-anchor=\"middle\" font-size=\"14\">" << aux_escape(xlabel) << "</text>\n"
        << "<text transform=\"translate(16 " << top + ph / 2 << ") rotate(-90)\" "
        << "text-anchor=\"middle\" font-size=\"14\">" << aux_escape(ylabel) << "</text>\n";
    // The title, if there is one.
    if (!title.empty())
        out << "<text x=\"" << left + pw / 2 << "\" y=\"22\" text-anchor=\"middle\" "
            << "font-size=\"15\">" << aux_escape(title) << "</text>\n";

    // The legend: one row per labelled layer, in the upper right corner.
    double row = top + 16;
    for (const Layer &l : layers_) {
        if (l.label.empty()) continue;
        const Color swatch = l.filled || l.kind == Kind::Region ? l.face : l.edge;
        out << "<rect x=\"" << left + pw - 130 << "\" y=\"" << row - 10
            << "\" width=\"14\" height=\"10\""
            << " fill=\"" << swatch.hex() << "\" stroke=\"" << l.edge.hex() << "\"/>\n<text x=\""
            << left + pw - 110 << "\" y=\"" << row - 1 << "\">" << aux_escape(l.label)
            << "</text>\n";
        row += 16;
    }
    out << "</svg>\n";
    return out.str();
}

void Figure::save(const std::string &path) const {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Figure::save: cannot write '" + path + "'");
    file << svg();
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
