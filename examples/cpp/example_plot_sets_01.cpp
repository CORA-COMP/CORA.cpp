// example_plot_sets_01 - a set in one, two and three dimensions
//
// plot draws the same set projected onto one dimension (a bar), two (a polygon) or three (the
// faces of a polytope), for every set class. The core decides how it looks; SVG is one writer.
//
// Syntax:   build/examples/cpp/example_plot_sets_01
// Outputs:  example_plot_sets_01_1d.svg, _2d.svg and _3d.svg
// See also: example_plot_sets_01 of the Python examples, example_plot_tikz_01_group

#include "cora/global/plot/plot.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    // Sets ------------------------------------------------------------------------------------

    const Zonotope Z(Tensor({0, 0, 0}),
                     Tensor({{1, 0, 0.5, 0.2}, {0, 1, 0.5, -0.3}, {0, 0, 1, 0.4}}));
    const Interval I(Tensor({-1.5, -1, -0.5}), Tensor({0.5, 1.5, 2}));

    // One Dimension ---------------------------------------------------------------------------

    Figure bars;
    plot(bars, Z, {0}, {.label = "zonotope", .line_width = 4});
    plot(bars, I, {0}, {.label = "interval", .line_width = 4});
    bars.save("example_plot_sets_01_1d.svg");

    // Two Dimensions --------------------------------------------------------------------------

    Figure polygons;
    plot(polygons, Z, {0, 1}, {.label = "zonotope"});
    plot(polygons, I, {0, 1}, {.label = "interval", .filled = false});
    polygons.title = "two dimensions";
    polygons.save("example_plot_sets_01_2d.svg");

    // Three Dimensions ------------------------------------------------------------------------

    // The faces come from the convex hull of the vertices, computed once in the core.
    Figure faces;
    plot(faces, Z, {0, 1, 2}, {.label = "zonotope"});
    plot(faces, I, {0, 1, 2}, {.label = "interval", .face_alpha = 0.1});
    faces.save("example_plot_sets_01_3d.svg");
    std::cout << "faces of the zonotope: " << faces.layers()[0].faces.size() << "\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
