// svg - a figure, or a grid of figures, written as SVG
//
// Syntax:   std::string text = figure().svg();   figure().save("figure.svg");
// Inputs:   the layers of the figure, stacked by their zorder
// Outputs:  the SVG text, or the file
// See also: cora/global/plot/figure.h, src/global/plot/figure.cpp

#include "cora/global/error.h"
#include "cora/global/plot/figure.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Text and Ticks ----------------------------------------------------------------------------------

// A number with up to four significant digits and no trailing zeros.
std::string aux_num(double v) {
    char out[32];
    std::snprintf(out, sizeof out, "%.4g", std::abs(v) < numerics::kPrintZero ? 0.0 : v);
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
    for (double t = std::ceil(lo / step) * step; t <= hi + numerics::kTickSlack * step; t += step)
        ticks.push_back(t);
    return ticks;
}

// Layers ------------------------------------------------------------------------------------------

// The drawing area of an axes inside its cell, and how a point of the figure maps into it.
struct Frame {
    double left, top, pw, ph;
    Figure::Limits lim;
    double X(double x) const { return left + (x - lim.x0) / (lim.x1 - lim.x0) * pw; }
    double Y(double y) const { return top + (lim.y1 - y) / (lim.y1 - lim.y0) * ph; }
};

// The path of a polygon in the frame, closed or not.
std::string aux_path(const Frame &f, const Polygon &p, bool close) {
    std::string d;
    for (std::size_t i = 0; i < p.size(); ++i)
        d += (i ? " L" : "M") + aux_num(f.X(p[i][0])) + " " + aux_num(f.Y(p[i][1]));
    return d + (close ? " Z" : "");
}

// The layers, stacked by zorder (regions at the bottom), those of one zorder as added.
std::vector<const Figure::Layer *> aux_stacked(const std::vector<Figure::Layer> &layers) {
    std::vector<const Figure::Layer *> stacked;
    stacked.reserve(layers.size());
    for (const Figure::Layer &l : layers) stacked.push_back(&l);
    std::stable_sort(
        stacked.begin(), stacked.end(),
        [](const Figure::Layer *p, const Figure::Layer *q) { return p->zorder < q->zorder; });
    return stacked;
}

// One layer of a two-dimensional figure.
void aux_layer(std::ostringstream &out, const Frame &f, const Figure::Layer &l) {
    switch (l.kind) {
    case Figure::Kind::region: {
        const Figure::Limits &lim = f.lim;
        const Polygon box = {
            {lim.x0, lim.y0}, {lim.x1, lim.y0}, {lim.x1, lim.y1}, {lim.x0, lim.y1}};
        const Polygon r = Figure::clip_region(box, l.a, l.b, l.sign);
        if (r.size() >= 3)
            out << "<path d=\"" << aux_path(f, r, true) << "\" fill=\"" << l.face.hex()
                << "\" stroke=\"" << l.edge.hex() << "\" stroke-width=\"1.2\"/>\n";
        break;
    }
    case Figure::Kind::polygons: {
        std::string d; // all polygons in one path: its fill is their union
        for (const Polygon &p : l.polygons)
            if (p.size() >= 2) d += aux_path(f, p, true) + " ";
        for (const Polygon &p : l.polygons)
            if (p.size() == 1)
                out << "<circle cx=\"" << aux_num(f.X(p[0][0])) << "\" cy=\""
                    << aux_num(f.Y(p[0][1])) << "\" r=\"1.5\" fill=\"" << l.edge.hex() << "\"/>\n";
        if (d.empty()) break;
        out << "<path d=\"" << d << "\" fill=\"" << (l.filled ? l.face.hex() : std::string("none"))
            << "\" fill-opacity=\"" << (l.filled ? l.face_alpha : 1.0) << "\" stroke=\""
            << l.edge.hex() << "\" stroke-width=\"" << l.line_width
            << "\" stroke-linejoin=\"miter\"" << (l.dashed ? " stroke-dasharray=\"2 2\"" : "")
            << "/>\n";
        break;
    }
    // A line through the points, without fill.
    case Figure::Kind::polyline:
        out << "<path d=\"" << aux_path(f, l.polygons[0], false) << "\" fill=\"none\" stroke=\""
            << l.edge.hex() << "\" stroke-width=\"" << l.line_width << "\"/>\n";
        break;
    // A dot per point.
    case Figure::Kind::points:
        for (const Point &p : l.polygons[0])
            out << "<circle cx=\"" << aux_num(f.X(p[0])) << "\" cy=\"" << aux_num(f.Y(p[1]))
                << "\" r=\"" << l.radius << "\" fill=\"" << l.edge.hex() << "\"/>\n";
        break;
    default:
        CORA_FAIL(Error::report_to_dev, "Figure::svg: unknown layer kind; this is a bug in Figure");
    }
}

// The faces of the surfaces, far ones first, in the picture plane of the view.
void aux_surfaces(std::ostringstream &out, const Figure &fig, const Frame &f) {
    struct Face {
        double depth;
        const Figure::Layer *layer;
        Polygon shape;
    };
    // Each face with its mean depth, so that the far ones are drawn first.
    std::vector<Face> faces;
    for (const Figure::Layer &l : fig.layers())
        for (const std::vector<Point3> &face : l.faces) {
            double depth = 0;
            Face item{0, &l, {}};
            for (const Point3 &p : face) {
                double d = 0;
                const Point v = fig.view(p, &d);
                item.shape.push_back(v);
                depth += d / static_cast<double>(face.size());
            }
            item.depth = depth;
            faces.push_back(std::move(item));
        }
    std::stable_sort(faces.begin(), faces.end(),
                     [](const Face &a, const Face &b) { return a.depth < b.depth; });
    // The layer decides how a face looks; the faces of one layer share it.
    for (const Face &item : faces) {
        const Figure::Layer &l = *item.layer;
        out << "<path d=\"" << aux_path(f, item.shape, true) << "\" fill=\""
            << (l.filled ? l.face.hex() : std::string("none")) << "\" fill-opacity=\""
            << (l.filled ? l.face_alpha : 1.0) << "\" stroke=\"" << l.edge.hex()
            << "\" stroke-width=\"" << l.line_width << "\" stroke-linejoin=\"round\"/>\n";
    }
}

// Axes --------------------------------------------------------------------------------------------

// The frame, ticks and axis titles of a two-dimensional figure; the labels of an axis are left
// out if `x_ticks` or `y_ticks` is false (a cell of a grid that shares that axis).
void aux_axes(std::ostringstream &out, const Figure &fig, const Frame &f, double height,
              bool x_ticks, bool y_ticks) {
    const double left = f.left, top = f.top, pw = f.pw, ph = f.ph;
    out << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << pw << "\" height=\"" << ph
        << "\" fill=\"none\" stroke=\"black\"/>\n";
    // The ticks of the x axis, with their numbers unless a grid shares them.
    for (double t : aux_ticks(f.lim.x0, f.lim.x1)) {
        out << "<line x1=\"" << aux_num(f.X(t)) << "\" y1=\"" << top + ph << "\" x2=\""
            << aux_num(f.X(t)) << "\" y2=\"" << top + ph + 5 << "\" stroke=\"black\"/>\n";
        if (x_ticks)
            out << "<text x=\"" << aux_num(f.X(t)) << "\" y=\"" << top + ph + 19
                << "\" text-anchor=\"middle\">" << aux_num(t) << "</text>\n";
    }
    // The same for the y axis, on the left.
    for (double t : aux_ticks(f.lim.y0, f.lim.y1)) {
        out << "<line x1=\"" << left - 5 << "\" y1=\"" << aux_num(f.Y(t)) << "\" x2=\"" << left
            << "\" y2=\"" << aux_num(f.Y(t)) << "\" stroke=\"black\"/>\n";
        if (y_ticks)
            out << "<text x=\"" << left - 9 << "\" y=\"" << aux_num(f.Y(t) + 4)
                << "\" text-anchor=\"end\">" << aux_num(t) << "</text>\n";
    }
    // The titles of the axes.
    if (x_ticks)
        out << "<text x=\"" << left + pw / 2 << "\" y=\"" << height - 12
            << "\" text-anchor=\"middle\" font-size=\"14\">" << aux_escape(fig.xlabel)
            << "</text>\n";
    if (y_ticks)
        out << "<text transform=\"translate(16 " << top + ph / 2 << ") rotate(-90)\" "
            << "text-anchor=\"middle\" font-size=\"14\">" << aux_escape(fig.ylabel) << "</text>\n";
}

// The legend: one row per labelled layer, in the upper right corner.
void aux_legend(std::ostringstream &out, const Figure &fig, const Frame &f) {
    double row = f.top + 16;
    for (const Figure::Layer &l : fig.layers()) {
        if (l.label.empty()) continue;
        const Color swatch = l.filled || l.kind == Figure::Kind::region ? l.face : l.edge;
        out << "<rect x=\"" << f.left + f.pw - 130 << "\" y=\"" << row - 10
            << "\" width=\"14\" height=\"10\""
            << " fill=\"" << swatch.hex() << "\" fill-opacity=\"" << (l.filled ? l.face_alpha : 1.0)
            << "\" stroke=\"" << l.edge.hex() << "\"/>\n<text x=\"" << f.left + f.pw - 110
            << "\" y=\"" << row - 1 << "\">" << aux_escape(l.label) << "</text>\n";
        row += 16;
    }
}

// The wire frame of the limits box of a three-dimensional figure, and the names of the axes.
void aux_box(std::ostringstream &out, const Figure &fig, const Frame &f, const Figure::Limits &l) {
    auto at = [&](double x, double y, double z) { return fig.view({x, y, z}); };
    for (int k = 0; k < 8; ++k)
        for (int axis = 0; axis < 3; ++axis) {
            if (k & (1 << axis)) continue; // each of the 12 edges once, from its lower end
            Point3 a = {k & 1 ? l.x1 : l.x0, k & 2 ? l.y1 : l.y0, k & 4 ? l.z1 : l.z0}, b = a;
            b[static_cast<std::size_t>(axis)] = axis == 0 ? l.x1 : axis == 1 ? l.y1 : l.z1;
            out << "<path d=\"" << aux_path(f, {at(a[0], a[1], a[2]), at(b[0], b[1], b[2])}, false)
                << "\" fill=\"none\" stroke=\"#bbbbbb\"/>\n";
        }
    const std::string names[3] = {fig.xlabel, fig.ylabel, fig.zlabel};
    const Point ends[3] = {at(l.x1, l.y0, l.z0), at(l.x1, l.y1, l.z0), at(l.x0, l.y0, l.z1)};
    for (int axis = 0; axis < 3; ++axis)
        out << "<text x=\"" << aux_num(f.X(ends[axis][0])) << "\" y=\""
            << aux_num(f.Y(ends[axis][1]) + 14) << "\" text-anchor=\"middle\" font-size=\"14\">"
            << aux_escape(names[axis]) << "</text>\n";
}

// The axes of a figure of three dimensions: the box and the surfaces, scaled to fit the area, with
// the same scale on both axes of the picture plane.
void aux_three_dimensional(std::ostringstream &out, const Figure &fig, Frame f) {
    const Figure::Limits l = f.lim;
    // The extent of the box in the picture plane gives the scale, equal on both axes.
    double u0 = 1e300, u1 = -1e300, v0 = 1e300, v1 = -1e300;
    for (int k = 0; k < 8; ++k) {
        const Point v = fig.view({k & 1 ? l.x1 : l.x0, k & 2 ? l.y1 : l.y0, k & 4 ? l.z1 : l.z0});
        u0 = std::min(u0, v[0]);
        u1 = std::max(u1, v[0]);
        v0 = std::min(v0, v[1]);
        v1 = std::max(v1, v[1]);
    }
    const double s = std::min(f.pw / (u1 - u0), f.ph / (v1 - v0));
    const double pad_u = (f.pw / s - (u1 - u0)) / 2, pad_v = (f.ph / s - (v1 - v0)) / 2;
    f.lim = {u0 - pad_u, u1 + pad_u, v0 - pad_v, v1 + pad_v, 0, 0};
    aux_box(out, fig, f, l);
    aux_surfaces(out, fig, f);
}

// Documents ---------------------------------------------------------------------------------------

// The group of one figure in the cell at (x, y) of size width x height. `lim` are the
// limits to draw with, the figure's own unless the grid shares them.
std::string aux_figure(const Figure &fig, double x, double y, double width, double height,
                       const Figure::Limits &lim, bool x_ticks, bool y_ticks) {
    const double left = 70, right = 20, top = fig.title.empty() ? 20 : 40, bottom = 55;
    Frame f{left, top, width - left - right, height - top - bottom, lim};
    std::ostringstream out;
    out << "<g transform=\"translate(" << x << " " << y << ")\">\n";
    if (fig.dimension == 3) {
        aux_three_dimensional(out, fig, f);
    } else {
        // The layers are cut at the frame of the axes.
        out << "<clipPath id=\"area" << x << "_" << y << "\"><rect x=\"" << left << "\" y=\"" << top
            << "\" width=\"" << f.pw << "\" height=\"" << f.ph
            << "\"/></clipPath>\n<g clip-path=\"url(#area" << x << "_" << y << ")\">\n";
        for (const Figure::Layer *layer : aux_stacked(fig.layers())) aux_layer(out, f, *layer);
        out << "</g>\n";
    }
    if (fig.dimension != 3) aux_axes(out, fig, f, height, x_ticks, y_ticks);
    if (!fig.title.empty())
        out << "<text x=\"" << left + f.pw / 2 << "\" y=\"22\" text-anchor=\"middle\" "
            << "font-size=\"15\">" << aux_escape(fig.title) << "</text>\n";
    aux_legend(out, fig, f);
    out << "</g>\n";
    return out.str();
}

// The opening of the document: the size, the font and a white background.
std::string aux_root(double width, double height) {
    std::ostringstream out;
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height
        << "\" viewBox=\"0 0 " << width << " " << height << "\" font-family=\"sans-serif\" "
        << "font-size=\"12\">\n<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    return out.str();
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::string Figure::svg() const {
    const double top = title.empty() ? 20 : 40;
    const Limits lim = limits(width - 90, height - top - 55);
    return aux_root(width, height) + aux_figure(*this, 0, 0, width, height, lim, true, true) +
           "</svg>\n";
}

void Figure::save(const std::string &path) const { write_text_file(path, svg(), "Figure::save"); }

// The cells side by side in one picture; a shared axis has the union of the limits of its cells
// and its numbers and title only at the last row or the first column.
std::string FigureGrid::svg() const {
    const double cw = width / cols_, ch = height / rows_;
    Figure::Limits shared = at(0, 0).limits();
    for (const Figure &f : cells_) {
        const Figure::Limits l = f.limits();
        shared.x0 = std::min(shared.x0, l.x0);
        shared.x1 = std::max(shared.x1, l.x1);
        shared.y0 = std::min(shared.y0, l.y0);
        shared.y1 = std::max(shared.y1, l.y1);
    }
    std::ostringstream out;
    out << aux_root(width, height);
    // Cell by cell, each with its own limits unless the axis is shared.
    for (int r = 0; r < rows_; ++r)
        for (int c = 0; c < cols_; ++c) {
            const Figure &f = at(r, c);
            const double top = f.title.empty() ? 20 : 40;
            Figure::Limits lim = f.limits(cw - 90, ch - top - 55);
            if (share_x) {
                lim.x0 = shared.x0;
                lim.x1 = shared.x1;
            }
            if (share_y) {
                lim.y0 = shared.y0;
                lim.y1 = shared.y1;
            }
            // The numbers of a shared axis are written at the last row and the first column.
            out << aux_figure(f, c * cw, r * ch, cw, ch, lim, !share_x || r == rows_ - 1,
                              !share_y || c == 0);
        }
    out << "</svg>\n";
    return out.str();
}

void FigureGrid::save(const std::string &path) const {
    write_text_file(path, svg(), "FigureGrid::save");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
