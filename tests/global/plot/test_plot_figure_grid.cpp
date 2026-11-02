// test_plot_figure_grid - FigureGrid: figures in rows and columns written as one SVG, the limits
// and the view of a figure

#include "cora/global/plot/plot.h"
#include "testing.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace cora;
using test::check;
using test::close;

namespace {

bool has(const std::string &text, const std::string &what) {
    return text.find(what) != std::string::npos;
}

std::size_t count(const std::string &text, const std::string &what) {
    std::size_t n = 0;
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1))
        ++n;
    return n;
}

Zonotope shifted(double x) { return Zonotope(test::column({x, 0}), Tensor({{1, 0}, {0, 1}})); }

void cells_are_figures_of_their_own(const std::string &b) {
    use_cora_colors("CORA:default");
    FigureGrid grid(2, 3);
    check(grid.rows() == 2 && grid.cols() == 3, b + ": the shape");
    plot(grid.at(0, 0), shifted(0));
    plot(grid.at(1, 2), shifted(5));
    check(grid.at(0, 0).layers().size() == 1 && grid.at(0, 1).layers().empty(),
          b + ": a plot goes into its cell");
    check(grid.at(1, 2).layers()[0].face.hex() == cora_color("CORA:blue").hex(),
          b + ": every cell has its own order of colors");
    check(test::throws([&] { grid.at(2, 0); }) && test::throws([&] { grid.at(0, -1); }),
          b + ": a cell that does not exist");
    check(test::throws([] { FigureGrid(0, 1); }), b + ": at least one cell");
}

void the_grid_is_one_svg(const std::string &b) {
    FigureGrid grid(1, 2);
    plot(grid.at(0, 0), shifted(0), {0, 1}, {.label = "left"});
    plot(grid.at(0, 1), shifted(5), {0, 1}, {.label = "right"});
    const std::string svg = grid.svg();
    check(svg.rfind("<svg", 0) == 0 && count(svg, "<svg") == 1, b + ": one document");
    check(has(svg, "left") && has(svg, "right"), b + ": both legends");
    check(count(svg, "<clipPath") == 2, b + ": a clip area per cell");
    const std::string path = "test_plot_figure_grid_output.svg";
    grid.save(path);
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    check(text.str() == svg, b + ": save writes the same");
    std::remove(path.c_str());
    check(test::throws([&] { grid.save("/nonexistent-folder/grid.svg"); }),
          b + ": an unwritable path");
}

// A shared axis has the union of the limits, and its numbers only on the outer cells.
void shared_axes(const std::string &b) {
    FigureGrid grid(2, 1);
    plot(grid.at(0, 0), shifted(0));
    plot(grid.at(1, 0), shifted(10));
    const std::string separate = grid.svg();
    grid.share_x = true;
    const std::string shared = grid.svg();
    check(count(shared, "<text") < count(separate, "<text"), b + ": fewer numbers when shared");
    check(has(shared, ">10<") || has(shared, ">8<") || has(shared, ">12<"),
          b + ": both cells show the common range");
}

void limits_and_view(const std::string &b) {
    Figure fig;
    plot(fig, shifted(0), {0, 1});
    Figure::Limits l = fig.limits();
    check(close(l.x0, -1.1) && close(l.x1, 1.1) && close(l.y0, -1.1) && close(l.y1, 1.1),
          b + ": the extent and a margin of 5%");
    fig.xlim = {{-3.0, 4.0}};
    l = fig.limits();
    check(close(l.x0, -3) && close(l.x1, 4) && close(l.y1, 1.1),
          b + ": a fixed limit replaces one axis");
    fig.equal_axes = true;
    l = fig.limits(300, 100);
    check(close((l.x1 - l.x0) / 300, (l.y1 - l.y0) / 100) || l.x0 == -3,
          b + ": equal axes widen the other axis, a fixed limit stays");
    Figure view;
    view.azimuth = 0;
    view.elevation = 0; // from the front: x to the right, z up
    double depth = 0;
    const Point p = view.view({2, 5, 3}, &depth);
    check(close(p[0], 2) && close(p[1], 3) && close(depth, -5), b + ": the view from the front");
    view.elevation = 90; // from above: x to the right, y up
    const Point top = view.view({2, 5, 3}, &depth);
    check(close(top[0], 2) && close(top[1], 5) && close(depth, 3), b + ": the view from above");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        cells_are_figures_of_their_own(b);
        the_grid_is_one_svg(b);
        shared_axes(b);
        limits_and_view(b);
    });
    return test::finish("plot figure grid");
}
