// tikz - a figure, or a grid of figures, written as pgfplots code
//
// A grid gives a groupplot; with external_data the tables of the coordinates are written.
//
// Syntax:   std::string text = figure().tikz({.standalone = true});   figure().save_tikz("a.tikz");
// Inputs:   layers - the layers of the figure, stacked by their zorder
//           options - the TikzOptions
// Outputs:  text - a tikzpicture with one \addplot per layer, or a LaTeX document around it
// See also: cora/global/plot/tikz.h, cora/global/plot/figure.h, src/global/plot/svg.cpp
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/plot/figure.h"
#include "cora/global/plot/geometry.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Colors ------------------------------------------------------------------------------------------

// The colors that cora-macs names, with its values (experiments/paper/latex/cora-macs.sty of
// CORALean), one per line: the macro without "CORAcolor", the CORA identifier, r g b. A layer has
// such a color if it is that of the CORA identifier. The file defines UnsafeLight and Highlight2
// with other values than CORA's CORAcolor.m, and a standalone file follows cora-macs.
struct NamedColor {
    std::string macro, identifier;
    Color rgb;
};

const std::vector<NamedColor> &aux_named_colors() {
    static const std::vector<NamedColor> all = [] {
        std::istringstream table(
            "ReachSet reachSet 0.2706 0.5882 1\n InitialSet initialSet 1 1 1\n"
            "FinalSet finalSet 0.9 0.9 0.9\n Simulations simulations 0 0 0\n"
            "Unsafe unsafe 0.9451 0.5529 0.5686\n UnsafeLight unsafeLight 0.9882 0.9098 0.9137\n"
            "Safe safe 0.4706 0.7725 0.4980\n Blue blue 0 0.4470 0.7410\n"
            "Red red 0.8500 0.3250 0.0980\n Yellow yellow 0.9290 0.6940 0.1250\n"
            "Purple purple 0.4940 0.1840 0.5560\n Green green 0.4660 0.6740 0.1880\n"
            "LightBlue light-blue 0.3010 0.7450 0.9330\n DarkRed dark-red 0.6350 0.0780 0.1840\n"
            "Highlight1 highlight1 1 0.6824 0.2980\n Highlight2 highlight2 0.3804 0.6275 1\n");
        // Each row is read into a color, with the names of CORA's and cora-macs's spelling.
        std::vector<NamedColor> colors;
        NamedColor c;
        while (table >> c.macro >> c.identifier >> c.rgb.r >> c.rgb.g >> c.rgb.b) {
            c.macro = "CORAcolor" + c.macro;
            c.identifier = "CORA:" + c.identifier;
            colors.push_back(c);
        }
        return colors;
    }();
    return all;
}

bool aux_same_color(const Color &a, const Color &b) {
    return std::abs(a.r - b.r) < numerics::kColorEqual &&
           std::abs(a.g - b.g) < numerics::kColorEqual &&
           std::abs(a.b - b.b) < numerics::kColorEqual;
}

// Numbers -----------------------------------------------------------------------------------------

// A number with the decimals given, without trailing zeros (so 0.50 is 0.5 and -0.0 is 0).
std::string aux_fixed(double v, int decimals) {
    char out[64];
    std::snprintf(out, sizeof out, "%.*f", decimals, v);
    std::string s = out;
    if (s.find('.') != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
    }
    return s == "-0" ? "0" : s;
}

// What is shared while one figure is written: the options, the colors used, the tables written.
struct Writer {
    const TikzOptions &options;
    std::map<std::string, Color> used; // the cora-macs colors the text names, for \definecolor
    std::string table_prefix;          // external data: the path of the tables without ".dat"
    std::vector<std::pair<std::string, std::string>> *tables;
    int plots = 0; // the \addplot commands with a table so far

    // A color as pgfplots reads it: its name in cora-macs, or its RGB value.
    std::string color(const Color &c) {
        if (options.coramacs)
            for (const NamedColor &named : aux_named_colors())
                if (aux_same_color(c, cora_color(named.identifier))) {
                    used[named.macro] = named.rgb;
                    return named.macro;
                }
        return "{rgb,1:red," + aux_fixed(c.r, 4) + ";green," + aux_fixed(c.g, 4) + ";blue," +
               aux_fixed(c.b, 4) + "}";
    }
};

// The decimals that give a hundred-thousandth of the range of an axis.
int aux_decimals(double lo, double hi) {
    const double range = std::max(hi - lo, numerics::kTiny);
    return std::clamp(static_cast<int>(std::ceil(-std::log10(range))) + 5, 0, 15);
}

// The limits of the axes and the decimals that its numbers are written with.
struct Axes {
    Figure::Limits lim;
    int dx, dy, dz;
};

Axes aux_axes(const Figure::Limits &lim) {
    return {lim, aux_decimals(lim.x0, lim.x1), aux_decimals(lim.y0, lim.y1),
            aux_decimals(lim.z0, lim.z1)};
}

// Layers ------------------------------------------------------------------------------------------

using Path = std::vector<std::array<double, 3>>;

// The options of a layer that is drawn as a filled or open shape.
std::string aux_style(Writer &w, const Figure::Layer &l) {
    std::vector<std::string> o;
    const bool cora_fill = w.options.coramacs && l.filled && aux_same_color(l.edge, l.face) &&
                           std::abs(l.face_alpha - 0.2) < numerics::kPlotEqual;
    if (cora_fill) {
        o.push_back("plot filled=" + w.color(l.face));
    } else {
        o.push_back("draw=" + w.color(l.edge));
        if (l.filled) {
            o.push_back("fill=" + w.color(l.face));
            if (l.face_alpha < 1) o.push_back("fill opacity=" + aux_fixed(l.face_alpha, 4));
            if (!l.label.empty()) o.emplace_back("area legend");
        }
    }
    if (std::abs(l.line_width - 1) > numerics::kPlotEqual)
        o.push_back("line width=" + aux_fixed(0.4 * l.line_width, 3) + "pt");
    if (l.dashed) o.emplace_back("dashed");
    std::string text;
    for (std::size_t i = 0; i < o.size(); ++i) text += (i ? ", " : "") + o[i];
    return text;
}

// The path of a polygon as the picture shows it: thinned, and closed by its first point.
Path aux_path(const Polygon &p, bool closed, const Axes &a, const TikzOptions &o) {
    const Polygon shown = o.thin ? thin_polygon(p,
                                                {o.thin_tolerance * (a.lim.x1 - a.lim.x0),
                                                 o.thin_tolerance * (a.lim.y1 - a.lim.y0)},
                                                closed)
                                 : p;
    Path path;
    for (const Point &v : shown) path.push_back({v[0], v[1], 0});
    if (closed && shown.size() >= 3) path.push_back(path.front());
    return path;
}

// The numbers of a point with `sep` between them: a space in a table, a comma in coordinates.
std::string aux_coordinate(const std::array<double, 3> &p, bool three, const Axes &a,
                           const char *sep) {
    return aux_fixed(p[0], a.dx) + sep + aux_fixed(p[1], a.dy) +
           (three ? sep + aux_fixed(p[2], a.dz) : "");
}

// The \addplot command for the paths, which pgfplots draws as one path (a jump separates them):
// inline as coordinates for one path, else as a table; with external data, a table in a file.
void aux_addplot(std::ostringstream &out, Writer &w, const std::string &options,
                 const std::vector<Path> &paths, bool three, const Axes &a,
                 const std::string &label) {
    const std::string cmd = three ? "\\addplot3" : "\\addplot";
    const std::string names = three ? "x y z" : "x y";
    const std::string jump = paths.size() > 1 ? ", unbounded coords=jump" : "";
    auto rows = [&](const std::string &separator) { // all coordinates, a jump between paths
        std::string text;
        for (std::size_t k = 0; k < paths.size(); ++k) {
            if (k) text += (three ? "nan nan nan" : "nan nan") + separator;
            for (const auto &p : paths[k]) text += aux_coordinate(p, three, a, " ") + separator;
        }
        return text;
    };
    // With external data the table goes to a file and the picture reads it.
    if (w.options.external_data) {
        const std::string file = w.table_prefix + "_" + std::to_string(++w.plots) + ".dat";
        if (w.tables) w.tables->emplace_back(file, names + "\n" + rows("\n"));
        out << "  " << cmd << "[" << options << jump << "] table[x=x, y=y" << (three ? ", z=z" : "")
            << "] {" << file << "};\n";
    } else if (paths.size() == 1) {
        // One path: inline coordinates, four to a line.
        out << "  " << cmd << "[" << options << "] coordinates {";
        for (std::size_t i = 0; i < paths[0].size(); ++i) {
            if (i % 4 == 0) out << "\n    ";
            out << (i % 4 ? " " : "") << "(" << aux_coordinate(paths[0][i], three, a, ",") << ")";
        }
        out << "\n  };\n";
    } else {
        // Several paths: an inline table, with a jump between them.
        out << "  " << cmd << "[" << options << jump << "] table[row sep=crcr] {\n    " << names
            << "\\\\\n";
        std::string text = rows("\\\\\n    ");
        out << "    " << text.substr(0, text.size() - 4) << "  };\n";
    }
    if (!label.empty()) out << "  \\addlegendentry{" << label << "}\n";
}

// A polygon layer: its loops (and its segments) as one path, its points as marks.
void aux_polygon_layer(std::ostringstream &out, Writer &w, const Figure::Layer &l, const Axes &a,
                       const std::string &options) {
    std::vector<Path> paths;
    std::vector<Path> dots;
    for (const Polygon &p : l.polygons) {
        if (p.size() == 1)
            dots.push_back(aux_path(p, false, a, w.options));
        else
            paths.push_back(aux_path(p, p.size() >= 3, a, w.options));
    }
    if (!paths.empty()) aux_addplot(out, w, options, paths, false, a, l.label);
    if (dots.empty()) return;
    Path all;
    for (const Path &d : dots) all.push_back(d[0]);
    aux_addplot(out, w, "only marks, mark=*, mark size=0.5pt, color=" + w.color(l.edge), {all},
                false, a, paths.empty() ? l.label : "");
}

// The faces of a surface, the far ones first, each its own \addplot3 (the legend names the layer
// once).
void aux_surface_layer(std::ostringstream &out, Writer &w, const Figure &fig,
                       const Figure::Layer &l, const Axes &a, const std::string &options) {
    // Each face with its mean depth, far ones first.
    std::vector<std::pair<double, const std::vector<Point3> *>> faces;
    for (const std::vector<Point3> &face : l.faces) {
        double depth = 0;
        for (const Point3 &p : face) {
            double d = 0;
            fig.view(p, &d);
            depth += d / static_cast<double>(face.size());
        }
        faces.emplace_back(depth, &face);
    }
    std::stable_sort(faces.begin(), faces.end(),
                     [](const auto &p, const auto &q) { return p.first < q.first; });
    // A face is a closed path; only the first carries the legend entry.
    bool first = true;
    for (const auto &entry : faces) {
        const std::vector<Point3> *face = entry.second;
        Path path;
        for (const Point3 &p : *face) path.push_back(p);
        if (face->size() >= 3) path.push_back(path.front());
        aux_addplot(out, w, options + (first ? "" : ", forget plot"), {path}, true, a,
                    first ? l.label : "");
        first = false;
    }
}

// All layers of a figure, stacked by zorder.
void aux_layers(std::ostringstream &out, Writer &w, const Figure &fig, const Axes &a) {
    std::vector<const Figure::Layer *> stacked;
    for (const Figure::Layer &l : fig.layers()) stacked.push_back(&l);
    std::stable_sort(
        stacked.begin(), stacked.end(),
        [](const Figure::Layer *p, const Figure::Layer *q) { return p->zorder < q->zorder; });
    for (const Figure::Layer *layer : stacked) {
        const Figure::Layer &l = *layer;
        switch (l.kind) {
        case Figure::Kind::region: {
            // A region is a polygon: the halfspace cut to the limits of the axes.
            const Polygon box = {{a.lim.x0, a.lim.y0},
                                 {a.lim.x1, a.lim.y0},
                                 {a.lim.x1, a.lim.y1},
                                 {a.lim.x0, a.lim.y1}};
            const Polygon r = Figure::clip_region(box, l.a, l.b, l.sign);
            if (r.size() < 3) break;
            Figure::Layer shape = l;
            shape.filled = true;
            shape.face_alpha = 1;
            aux_addplot(out, w, aux_style(w, shape), {aux_path(r, true, a, w.options)}, false, a,
                        l.label);
            break;
        }
        // The other kinds know their own way to be drawn.
        case Figure::Kind::polygons:
            aux_polygon_layer(out, w, l, a, aux_style(w, l));
            break;
        case Figure::Kind::surface:
            aux_surface_layer(out, w, fig, l, a, aux_style(w, l));
            break;
        // Lines and dots take the color of the layer's edge.
        case Figure::Kind::polyline:
            aux_addplot(out, w,
                        "draw=" + w.color(l.edge) +
                            (std::abs(l.line_width - 1) > numerics::kPlotEqual
                                 ? ", line width=" + aux_fixed(0.4 * l.line_width, 3) + "pt"
                                 : ""),
                        {aux_path(l.polygons[0], false, a, w.options)}, false, a, l.label);
            break;
        case Figure::Kind::points:
            aux_addplot(out, w,
                        "only marks, mark=*, mark size=" + aux_fixed(0.5 * l.radius, 2) +
                            "pt, color=" + w.color(l.edge),
                        {aux_path(l.polygons[0], false, a, w.options)}, false, a, l.label);
            break;
        default:
            CORA_FAIL(Error::report_to_dev,
                      "Figure::tikz: unknown layer kind; this is a bug in Figure");
        }
    }
}

// Axes --------------------------------------------------------------------------------------------

// The options of one axes: its texts, its limits, and what a group shares.
std::string aux_axis_options(const Figure &fig, const Axes &a, bool x_label, bool y_label) {
    std::vector<std::string> o;
    if (!fig.title.empty()) o.push_back("title={" + fig.title + "}");
    if (x_label && !fig.xlabel.empty()) o.push_back("xlabel={" + fig.xlabel + "}");
    if (y_label && !fig.ylabel.empty()) o.push_back("ylabel={" + fig.ylabel + "}");
    if (fig.dimension == 3 && !fig.zlabel.empty()) o.push_back("zlabel={" + fig.zlabel + "}");
    o.push_back("xmin=" + aux_fixed(a.lim.x0, a.dx) + ", xmax=" + aux_fixed(a.lim.x1, a.dx));
    o.push_back("ymin=" + aux_fixed(a.lim.y0, a.dy) + ", ymax=" + aux_fixed(a.lim.y1, a.dy));
    if (fig.dimension == 3) {
        o.push_back("zmin=" + aux_fixed(a.lim.z0, a.dz) + ", zmax=" + aux_fixed(a.lim.z1, a.dz));
        o.push_back("view={" + aux_fixed(fig.azimuth, 2) + "}{" + aux_fixed(fig.elevation, 2) +
                    "}");
    } else if (fig.equal_axes) {
        o.emplace_back("axis equal image");
    }
    std::string text;
    for (const std::string &s : o) text += "  " + s + ",\n";
    return text;
}

// The preamble of a standalone document: what the code needs, defined as cora-macs does.
std::string aux_preamble(const Writer &w, bool groups) {
    std::ostringstream out;
    out << "\\documentclass{article}\n\\usepackage[margin=1cm]{geometry}\n\\usepackage{pgfplots}\n";
    if (groups) out << "\\usepgfplotslibrary{groupplots}\n";
    out << "\\pgfplotsset{compat=1.18}\n\\pagestyle{empty}\n";
    for (const auto &[name, c] : w.used)
        out << "\\definecolor{" << name << "}{rgb}{" << aux_fixed(c.r, 4) << ","
            << aux_fixed(c.g, 4) << "," << aux_fixed(c.b, 4) << "}\n";
    if (w.options.coramacs)
        out << "\\pgfplotsset{plot filled/.style={area legend, draw=#1, fill=#1, fill "
               "opacity=0.2}}\n";
    return out.str() + "\\begin{document}\n";
}

// Documents ---------------------------------------------------------------------------------------

// A grid of figures (or one) as a tikzpicture, and as a document if asked.
std::string aux_tikz(const std::vector<const Figure *> &figures, int rows, int cols, bool share_x,
                     bool share_y, const TikzOptions &options, const std::string &table_prefix,
                     std::vector<std::pair<std::string, std::string>> *tables) {
    if (options.external_data && !tables)
        CORA_FAIL(Error::wrong_value, "tikz: external_data writes tables, so use save_tikz; tikz "
                                      "returns the text only");
    Writer w{options, {}, table_prefix, tables};
    const bool group = rows * cols > 1;
    // The limits of the cells; a shared axis has the union of the limits of all of them.
    std::vector<Figure::Limits> limits;
    limits.reserve(figures.size());
    for (const Figure *f : figures) limits.push_back(f->limits());
    for (Figure::Limits &l : limits)
        for (const Figure::Limits &o : limits) {
            if (share_x) {
                l.x0 = std::min(l.x0, o.x0);
                l.x1 = std::max(l.x1, o.x1);
            }
            if (share_y) {
                l.y0 = std::min(l.y0, o.y0);
                l.y1 = std::max(l.y1, o.y1);
            }
        }
    std::ostringstream body;
    body << "\\begin{tikzpicture}\n";
    // The size of every axes: as asked, else pgfplots's (in a group 5cm by 4cm).
    std::string size;
    if (!options.width.empty() || group)
        size += "  width=" + (options.width.empty() ? std::string("5cm") : options.width) + ",\n";
    if (!options.height.empty() || group)
        size +=
            "  height=" + (options.height.empty() ? std::string("4cm") : options.height) + ",\n";
    if (group) {
        body << "\\begin{groupplot}[\n  group style={\n    group size=" << cols << " by " << rows
             << ",\n    horizontal sep=1.6cm, vertical sep=1.6cm";
        if (share_x) body << ",\n    x descriptions at=edge bottom";
        if (share_y) body << ",\n    y descriptions at=edge left";
        body << "\n  },\n";
        body << size << "]\n";
    }
    // Each figure is an axes of a single picture, or the next one of the group.
    for (std::size_t k = 0; k < figures.size(); ++k) {
        const Figure &fig = *figures[k];
        const int r = static_cast<int>(k) / cols, c = static_cast<int>(k) % cols;
        const Axes a = aux_axes(limits[k]);
        const std::string opts =
            aux_axis_options(fig, a, !share_x || r == rows - 1, !share_y || c == 0);
        if (group)
            body << "\\nextgroupplot[\n" << opts << "]\n";
        else
            body << "\\begin{axis}[\n" << size << opts << "]\n";
        aux_layers(body, w, fig, a);
    }
    body << (group ? "\\end{groupplot}\n" : "\\end{axis}\n") << "\\end{tikzpicture}\n";
    if (options.standalone) return aux_preamble(w, group) + body.str() + "\\end{document}\n";
    return std::string("% pgfplots") + (group ? " with \\usepgfplotslibrary{groupplots}" : "") +
           "; CORA's colors and `plot filled` are those of cora-macs\n" + body.str();
}

// The path without its extension, where the tables go.
std::string aux_stem(const std::string &path) {
    const std::size_t dot = path.find_last_of('.'), slash = path.find_last_of("/\\");
    return dot != std::string::npos && (slash == std::string::npos || dot > slash)
               ? path.substr(0, dot)
               : path;
}

// Writes the text and the tables of an export.
void aux_save(const std::string &path, const std::string &who, const std::string &text,
              const std::vector<std::pair<std::string, std::string>> &tables) {
    write_text_file(path, text, who);
    for (const auto &[file, content] : tables) write_text_file(file, content, who);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// One figure is a picture of one axes.
std::string Figure::tikz(const TikzOptions &options) const {
    return aux_tikz({this}, 1, 1, false, false, options, "", nullptr);
}

// With the tables of its coordinates if the options ask for them.
void Figure::save_tikz(const std::string &path, const TikzOptions &options) const {
    std::vector<std::pair<std::string, std::string>> tables;
    const std::string text = aux_tikz({this}, 1, 1, false, false, options, aux_stem(path), &tables);
    aux_save(path, "Figure::save_tikz", text, tables);
}

// A grid of several figures is one groupplot.
std::string FigureGrid::tikz(const TikzOptions &options) const {
    std::vector<const Figure *> figures;
    figures.reserve(cells_.size());
    for (const Figure &f : cells_) figures.push_back(&f);
    return aux_tikz(figures, rows_, cols_, share_x, share_y, options, "", nullptr);
}

// The same with the tables.
void FigureGrid::save_tikz(const std::string &path, const TikzOptions &options) const {
    std::vector<const Figure *> figures;
    figures.reserve(cells_.size());
    for (const Figure &f : cells_) figures.push_back(&f);
    std::vector<std::pair<std::string, std::string>> tables;
    const std::string text =
        aux_tikz(figures, rows_, cols_, share_x, share_y, options, aux_stem(path), &tables);
    aux_save(path, "FigureGrid::save_tikz", text, tables);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
