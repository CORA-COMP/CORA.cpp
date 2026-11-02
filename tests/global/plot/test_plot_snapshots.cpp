// test_plot_snapshots - the layers (geometry within a tolerance) and the TikZ text (exactly) of a
// zonotope, a reachable set over time, a 2 x 2 group of axes and a three-dimensional zonotope

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/plot/plot.h"
#include "snapshot.h"

using namespace cora;
using test::check_snapshot;
using test::dump_layers;

namespace {

Zonotope initial_set() { return Zonotope(test::column({1, 0}), Tensor({{0.1, 0}, {0, 0.1}})); }

ReachSet oscillator() {
    return LinearSys(Tensor({{-0.1, 1}, {-1, -0.1}}))
        .reach({.initial_set = initial_set(), .t_final = 1.0},
               alg::Standard{.time_step = 0.2, .taylor_terms = 6});
}

Figure zonotope_figure() {
    use_cora_colors("CORA:default");
    Figure fig;
    const Zonotope Z(test::column({1, 0.5}), Tensor({{1, 0.3, 0}, {0, 0.5, 0.2}}));
    plot(fig, Z, {0, 1}, {.label = "$\\mathcal{Z}$"});
    fig.xlabel = "$x_1$";
    fig.ylabel = "$x_2$";
    return fig;
}

Figure over_time_figure() {
    use_cora_colors("CORA:contDynamics");
    Figure fig;
    plot_over_time(fig, oscillator(), 0, {.label = "reachable set"});
    fig.xlabel = "$t$";
    fig.ylabel = "$x_1$";
    use_cora_colors("CORA:default");
    return fig;
}

FigureGrid group_figure() {
    use_cora_colors("CORA:contDynamics");
    FigureGrid grid(2, 2);
    grid.share_x = true;
    const ReachSet R = oscillator();
    Figure &phase = grid.at(0, 0);
    plot(phase, Specification::unsafe_set(test::column({-1, 0}), -1.1), {0, 1},
         {.label = "unsafe"});
    plot(phase, R, {0, 1}, {.label = "reachable set"});
    plot(phase, initial_set(), {0, 1}, {.label = "initial set"});
    phase.xlabel = "$x_1$";
    phase.ylabel = "$x_2$";
    phase.title = "phase plane";
    plot_over_time(grid.at(0, 1), R, 0);
    grid.at(0, 1).ylabel = "$x_1$";
    plot_over_time(grid.at(1, 0), R, 1);
    grid.at(1, 0).xlabel = "$t$";
    grid.at(1, 0).ylabel = "$x_2$";
    const LinearSys sys(Tensor({{-0.1, 1}, {-1, -0.1}}));
    const std::vector<Tensor> x = sys.simulate(
        {.initial_state = test::column({1, 0}), .t_final = 1.0}, alg::Exact{.time_step = 0.2});
    plot_over_time(grid.at(1, 1), x, 0, 0.2, {.label = "simulation"});
    grid.at(1, 1).xlabel = "$t$";
    grid.at(1, 1).ylabel = "$x_1$";
    use_cora_colors("CORA:default");
    return grid;
}

Figure three_dimensional_figure() {
    use_cora_colors("CORA:default");
    Figure fig;
    const Zonotope Z(test::column({0, 0, 0}), Tensor({{1, 0, 0.5}, {0, 1, 0.5}, {0, 0, 1}}));
    plot(fig, Z, {0, 1, 2}, {.label = "Z"});
    fig.xlabel = "$x_1$";
    fig.ylabel = "$x_2$";
    fig.zlabel = "$x_3$";
    return fig;
}

void snapshots_of_the_layers() {
    const std::string here = __FILE__;
    check_snapshot(dump_layers(zonotope_figure()), test::snapshot_path(here, "zonotope.layers"),
                   "zonotope layers", 1e-9);
    check_snapshot(dump_layers(over_time_figure()), test::snapshot_path(here, "over_time.layers"),
                   "over time layers", 1e-9);
    const FigureGrid grid = group_figure();
    std::string all;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            all += "cell " + std::to_string(r) + " " + std::to_string(c) + "\n" +
                   dump_layers(grid.at(r, c));
    check_snapshot(all, test::snapshot_path(here, "group.layers"), "group layers", 1e-9);
    check_snapshot(dump_layers(three_dimensional_figure()),
                   test::snapshot_path(here, "zonotope_3d.layers"), "3-D layers", 1e-9);
}

void snapshots_of_the_tikz_text() {
    const std::string here = __FILE__;
    check_snapshot(zonotope_figure().tikz(), test::snapshot_path(here, "zonotope.tikz"),
                   "zonotope tikz");
    check_snapshot(over_time_figure().tikz(), test::snapshot_path(here, "over_time.tikz"),
                   "over time tikz");
    check_snapshot(group_figure().tikz(), test::snapshot_path(here, "group.tikz"), "group tikz");
    check_snapshot(three_dimensional_figure().tikz(), test::snapshot_path(here, "zonotope_3d.tikz"),
                   "3-D tikz");
    check_snapshot(group_figure().tikz({.standalone = true}),
                   test::snapshot_path(here, "group_standalone.tikz"), "standalone group tikz");
}

} // namespace

int main() {
    snapshots_of_the_layers();
    snapshots_of_the_tikz_text();
    return test::finish("plot snapshots");
}
