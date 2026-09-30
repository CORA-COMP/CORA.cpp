// figure - a two-dimensional figure: the layers plot() has drawn, written as SVG or handed on
//
// plot() puts layers into a figure with every choice made (colors, widths, order); the figure
// writes them as SVG, or a wrapper takes them out with takeLayers() and draws them itself (the
// Python plot draws them with matplotlib). Nothing is computed until then, so the order of the
// plot calls does not matter: layers are stacked by their zorder, the limits come from everything
// drawn, and the regions of a specification reach to the limits.
//
// Syntax:   plot(R, {0, 1});   figure().save("reach.svg");   Figure f;   plot(f, R, {0, 1});
// See also: global/plot/plot.h, plot/color.h

#pragma once

#include "contSet/contSet.h"
#include "global/plot/color.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Figure {
  public:
    enum class Kind { Polygons, Polyline, Points, Region };

    /// One thing drawn. `zorder` stacks layers: the higher ones lie on top (regions 0, reachable
    /// sets 1, their time points 2, sets 3, lines and points 4).
    struct Layer {
        Kind kind;
        std::vector<Polygon>
            polygons; // Polygons: the polygons; Polyline and Points: one, the points
        Color edge, face;
        bool filled = false, unify = true, dashed = false;
        double faceAlpha = 1;  // the opacity of the fill; the outline is opaque
        double lineWidth = 1, radius = 2, sign = 1, b = 0;
        Point a = {0, 0};
        std::string label = "";
        int zorder = 3;
    };

    /// Text of the figure; empty axis labels are set by plot() from the dimensions shown.
    std::string title, xlabel, ylabel;

    /// Whether one unit is as long on both axes.
    bool equalAxes = false;

    /// The size of the drawing in pixels.
    double width = 640, height = 480;

    /// Polygons: with `face` set they are filled, and (`unify`) drawn as their union, so that
    /// overlaps neither darken nor show inner edges; the outline follows the union's boundary.
    Layer &addPolygons(std::vector<Polygon> polygons, Color edge, std::optional<Color> face,
                       double lineWidth, const std::string &label, bool unify = true);

    /// A line through the points.
    Layer &addPolyline(std::vector<Point> points, Color color, double lineWidth,
                       const std::string &label);

    /// Dots at the points.
    Layer &addPoints(std::vector<Point> points, Color color, double radius,
                     const std::string &label);

    /// The region {x | sign * (a'x - b) >= 0}, shaded up to the limits of the figure.
    Layer &addRegion(Point a, double b, double sign, Color edge, Color face,
                     const std::string &label);

    /// The `k`-th color of CORA's order, k = 1, 2, ...: each call gives the next one, so
    /// successive plots differ (blue, red, yellow, ...). clear() starts again with the first.
    Color nextColor();

    /// The part of the polygon where sign * (a'x - b) >= 0, cut edge by edge: how a region
    /// is limited to the axes.
    static Polygon clipRegion(const Polygon &polygon, Point a, double b, double sign);

    /// The layers, in the order they were added; takeLayers hands them on and empties the figure
    /// of them (its texts and its color counter stay).
    const std::vector<Layer> &layers() const { return layers_; }
    std::vector<Layer> takeLayers();

    /// Removes everything.
    void clear();

    /// The figure as SVG text, and written to `path`.
    std::string svg() const;
    void save(const std::string &path) const;

  private:
    std::vector<Layer> layers_;
    int colorsGiven_ = 0;
};

/// The current figure, which plot() draws into.
Figure &figure();

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
