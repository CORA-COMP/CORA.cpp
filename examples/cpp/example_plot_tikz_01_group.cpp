// example_plot_tikz_01_group - four axes written as one pgfplots group plot
//
// A FigureGrid holds one figure per cell; its tikz() is a single groupplot with shared axes, CORA's
// colors by their names in cora-macs and thinned coordinates. The same text comes from Python.
//
// Syntax:   build/examples/cpp/example_plot_tikz_01_group
// Outputs:  example_plot_tikz_01_group.tikz (a tikzpicture) and _standalone.tikz (a document)
// See also: example_plot_tikz_01_group of the Python examples, example_plot_reach_01_over_time

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/plot/plot.h"

#include <iostream>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;

int main() {
    // Reachability Analysis -------------------------------------------------------------------

    const Zonotope X0(Tensor({1, 0}), Tensor({{0.1, 0}, {0, 0.1}}));
    const LinearSys sys(Tensor({{-0.1, 1}, {-1, -0.1}}));
    const ReachSet R = sys.reach({.initial_set = X0, .t_final = 3.0},
                                 alg::Standard{.time_step = 0.1, .taylor_terms = 6});

    // Figures ---------------------------------------------------------------------------------

    use_cora_colors("CORA:contDynamics");
    FigureGrid grid(2, 2);
    grid.share_x = true;

    // The phase plane in the first cell, the two states over time in the second and third.
    Figure &plane = grid.at(0, 0);
    plot(plane, R, {0, 1}, {.label = "reachable set"});
    plot(plane, X0, {0, 1}, {.label = "initial set"});
    plane.xlabel = "$x_1$";
    plane.ylabel = "$x_2$";
    plot_over_time(grid.at(0, 1), R, 0);
    grid.at(0, 1).ylabel = "$x_1$";
    plot_over_time(grid.at(1, 0), R, 1);
    grid.at(1, 0).xlabel = "$t$";
    grid.at(1, 0).ylabel = "$x_2$";

    // A three-dimensional set in the last one.
    const Zonotope Z(Tensor({0, 0, 0}), Tensor({{1, 0, 0.5}, {0, 1, 0.5}, {0, 0, 1}}));
    plot(grid.at(1, 1), Z, {0, 1, 2}, {.face_color = "CORA:blue"});
    grid.at(1, 1).xlabel = "$x_1$";
    grid.at(1, 1).ylabel = "$x_2$";
    grid.at(1, 1).zlabel = "$x_3$";

    // TikZ Export -----------------------------------------------------------------------------

    // A picture for \input into a paper that loads cora-macs, and a document that compiles alone.
    grid.save_tikz("example_plot_tikz_01_group.tikz");
    grid.save_tikz("example_plot_tikz_01_group_standalone.tikz", {.standalone = true});
    std::cout << grid.tikz().substr(0, 400) << "...\n";

    // example completed
    return 0;
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
