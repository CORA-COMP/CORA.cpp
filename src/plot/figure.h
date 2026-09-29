// figure - a two-dimensional figure that is written as SVG
//
// plot() draws into the current figure(); save() writes it. Nothing is computed until then, so
// the order of the plot calls does not matter: the limits come from everything drawn, and the
// regions of a specification reach to the limits.
//
// Syntax:   plot(R, {0, 1});   figure().save("reach.svg");
// See also: plot/plot.h, plot/color.h

#pragma once

#include "contSet/contSet.h"
#include "plot/color.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

class Figure {
  public:
    /// Text of the figure; empty axis labels are set by plot() from the dimensions shown.
    std::string title, xlabel, ylabel;

    /// Whether one unit is as long on both axes.
    bool equalAxes = false;

    /// The size of the drawing in pixels.
    double width = 640, height = 480;

    /// Polygons: with `face` set they are filled, and (`unify`) drawn as their union, so that
    /// overlaps neither darken nor show inner edges; the outline follows the union's boundary.
    void addPolygons(std::vector<Polygon> polygons, Color edge, std::optional<Color> face,
                     double lineWidth, const std::string &label, bool unify = true);

    /// A line through the points.
    void addPolyline(std::vector<Point> points, Color color, double lineWidth,
                     const std::string &label);

    /// Dots at the points.
    void addPoints(std::vector<Point> points, Color color, double radius, const std::string &label);

    /// The region {x | sign * (a'x - b) >= 0}, shaded up to the limits of the figure.
    void addRegion(Point a, double b, double sign, Color edge, Color face,
                   const std::string &label);

    /// The `k`-th color of the palette, k = 1, 2, ...: each call gives the next one.
    Color nextColor();

    /// Removes everything.
    void clear();

    /// The figure as SVG text, and written to `path`.
    std::string svg() const;
    void save(const std::string &path) const;

  private:
    enum class Kind { Polygons, Polyline, Points, Region };

    struct Layer {
        Kind kind;
        std::vector<Polygon>
            polygons; // Polygons: the polygons; Polyline and Points: one, the points
        Color edge, face;
        bool filled = false, unify = true;
        double lineWidth = 1, radius = 2, sign = 1, b = 0;
        Point a = {0, 0};
        std::string label = "";
    };

    std::vector<Layer> layers_;
    int colorsGiven_ = 0;
};

/// The current figure, which plot() draws into.
Figure &figure();

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
