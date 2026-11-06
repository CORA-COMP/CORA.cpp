// figure - one pair of axes: the layers plot() has drawn, written as SVG or TikZ or handed on
//
// plot() puts layers into a figure with every choice made (colors, widths, order, the union of
// overlapping sets); the figure writes them as SVG or pgfplots code, or a wrapper takes them out
// with layers() and draws them itself (the Python plot draws them with matplotlib). Nothing is
// computed until then, so the order of the plot calls does not matter: layers are stacked by
// their zorder, the limits come from everything drawn, and the regions of a specification reach
// to the limits. A FigureGrid is several figures in rows and columns (a pgfplots groupplot).
//
// Syntax:   plot(R, {0, 1});   figure().save("reach.svg");   figure().save_tikz("reach.tikz");
//           Figure f;   plot(f, R, {0, 1});   FigureGrid g(2, 2);   plot(g.at(0, 1), R);
// See also: cora/global/plot/plot.h, plot/color.h, plot/tikz.h
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/global/plot/color.h"
#include "cora/global/plot/tikz.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// One Axes ----------------------------------------------------------------------------------------

class Figure {
  public:
    enum class Kind { polygons, polyline, points, region, surface };

    /// One thing drawn. `zorder` stacks layers: the higher ones lie on top (regions 0, reachable
    /// sets 1, their time points 2, sets 3, lines and points 4).
    struct Layer {
        Kind kind;
        std::vector<Polygon>
            polygons; // Polygons: the polygons; Polyline and Points: one, the points
        std::vector<std::vector<Point3>> faces; // Surface: the faces of a polytope
        Color edge, face;
        bool filled = false, unify = true, dashed = false;
        double face_alpha = 1; // the opacity of the fill; the outline is opaque
        double line_width = 1, radius = 2, sign = 1, b = 0;
        Point a = {0, 0};
        std::string label = "";
        int zorder = 3;
    };

    /// Text of the figure, as LaTeX where it is written as TikZ; empty axis labels are set by
    /// plot() from the dimensions shown.
    std::string title, xlabel, ylabel, zlabel;

    /// Whether one unit is as long on both axes.
    bool equal_axes = false;

    /// 2, or 3 once a surface is drawn (plot of three dimensions); the two cannot be mixed.
    int dimension = 2;

    /// The view of a three-dimensional figure in degrees, as MATLAB's view(azimuth, elevation).
    double azimuth = -35, elevation = 30;

    /// The axis limits, if they are not those of the data (with a margin).
    std::optional<std::array<double, 2>> xlim, ylim, zlim;

    /// The size of the drawing in pixels.
    double width = 640, height = 480;

    /// Polygons: with `face` set they are filled. With `unify` they are replaced by their union
    /// (union_polygons), so that overlaps neither darken nor show inner edges.
    Layer &add_polygons(std::vector<Polygon> polygons, Color edge, std::optional<Color> face,
                        double line_width, const std::string &label, bool unify = true);

    /// A line through the points.
    Layer &add_polyline(std::vector<Point> points, Color color, double line_width,
                        const std::string &label);

    /// Dots at the points.
    Layer &add_points(std::vector<Point> points, Color color, double radius,
                      const std::string &label);

    /// The faces of a polytope in space, filled if `face` is set.
    Layer &add_surface(std::vector<std::vector<Point3>> faces, Color edge,
                       std::optional<Color> face, double line_width, const std::string &label);

    /// The region {x | sign * (a'x - b) >= 0}, shaded up to the limits of the figure.
    Layer &add_region(Point a, double b, double sign, Color edge, Color face,
                      const std::string &label);

    /// The `k`-th color of CORA's order, k = 1, 2, ...: each call gives the next one, so
    /// successive plots differ (blue, red, yellow, ...). clear() starts again with the first.
    Color next_color();

    /// The part of the polygon where sign * (a'x - b) >= 0, cut edge by edge: how a region
    /// is limited to the axes.
    static Polygon clip_region(const Polygon &polygon, Point a, double b, double sign);

    /// The layers, in the order they were added; take_layers hands them on and empties the figure
    /// of them (its texts and its color counter stay).
    const std::vector<Layer> &layers() const { return layers_; }
    std::vector<Layer> take_layers();

    /// The limits of the axes: `xlim` and so on, else the extent of the layers with a margin; with
    /// `equal_axes` and a drawing area of pw x ph (pw > 0), widened so that one unit is as long
    /// on both axes. The third pair is that of the z axis.
    struct Limits {
        double x0, x1, y0, y1, z0, z1;
    };
    Limits limits(double pw = 0, double ph = 0) const;

    /// A point of space in the picture plane of the view (azimuth, elevation), and its depth:
    /// the larger, the closer to the viewer.
    Point view(const Point3 &p, double *depth = nullptr) const;

    /// Removes everything.
    void clear();

    /// The figure as SVG text, and written to `path`.
    std::string svg() const;
    void save(const std::string &path) const;

    /// The figure as pgfplots code (see TikzOptions), and written to `path`.
    std::string tikz(const TikzOptions &options = {}) const;
    void save_tikz(const std::string &path, const TikzOptions &options = {}) const;

  private:
    std::vector<Layer> layers_;
    int colors_given_ = 0;
};

// Several Axes ------------------------------------------------------------------------------------

/// Figures in `rows` x `cols` cells, drawn as one picture: SVG cells, or one pgfplots groupplot.
class FigureGrid {
  public:
    FigureGrid(int rows, int cols);

    /// With `share_x` all cells have the same x limits (the union of theirs), and the x axis is
    /// labeled under the last row only; `share_y` likewise for y and the first column.
    bool share_x = false, share_y = false;

    /// The size of the whole drawing in pixels.
    double width = 640, height = 480;

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    /// The figure in cell (row, col), 0-based.
    Figure &at(int row, int col);
    const Figure &at(int row, int col) const;

    /// The grid as SVG text, written to `path`, as pgfplots code, and written to `path`.
    std::string svg() const;
    void save(const std::string &path) const;
    std::string tikz(const TikzOptions &options = {}) const;
    void save_tikz(const std::string &path, const TikzOptions &options = {}) const;

  private:
    int rows_, cols_;
    std::vector<Figure> cells_;
};

/// The current figure, which plot() draws into.
Figure &figure();

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
