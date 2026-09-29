// test_plot_svg - plot: sets, reachable sets, specifications and trajectories into an SVG figure

#include "contDynamics/linearSys/linearSys.h"
#include "global/rng.h"
#include "plot/plot.h"
#include "testing.h"

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace cora::ct;
using test::check;

namespace {

std::size_t count(const std::string &text, const std::string &what) {
    std::size_t n = 0;
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1))
        ++n;
    return n;
}

// The initial set, made on the backend that is current.
Zonotope initialSet() { return Zonotope(test::column({1, 0}), Tensor({{0.1, 0}, {0, 0.1}})); }

void a_set_is_an_initial_set_unless_told_otherwise(const std::string &b) {
    figure().clear();
    plot(initialSet(), {0, 1}, {.label = "initial set"});
    const std::string set = figure().svg();
    check(set.find("fill=\"#ffffff\"") != std::string::npos, b + ": white fill by default");
    check(set.find("stroke=\"#000000\"") != std::string::npos, b + ": black outline by default");
    check(set.find("initial set") != std::string::npos, b + ": the legend names the set");
    figure().clear();
    plot(initialSet(), {0, 1}, {.filled = false});
    check(figure().svg().find("fill-rule") == std::string::npos, b + ": open when not filled");
    figure().clear();
    plot(initialSet(), {0, 1}, {.facecolor = "CORA:red"});
    check(figure().svg().find(CORAcolor("CORA:red").hex()) != std::string::npos,
          b + ": a facecolor replaces the default");
}

void a_reach_set_is_drawn_as_one_union(const std::string &b) {
    figure().clear();
    const Reach R = LinearSys(Tensor({{-0.1, 1}, {-1, -0.1}})).reach(initialSet(), 0.1, 2.0, 6);
    plot(R, {0, 1}, {.label = "reachable set"});
    const std::string svg = figure().svg();
    // The union is one path for the outline and one for the fill, however many sets there are.
    check(count(svg, "fill-rule=\"nonzero\"") == 2, b + ": one region for all time intervals");
    check(figure().equalAxes, b + ": a reach plot has equal axes");
    figure().clear();
    plot(R, {0, 1}, {.unify = false});
    check(count(figure().svg(), "<path") >= 1, b + ": not unified still draws");
}

void a_specification_is_shaded_up_to_the_limits(const std::string &b) {
    figure().clear();
    plot(initialSet(), {0, 1});
    plot(Specification::unsafeSet(test::column({1, 0}), 0.5), {0, 1}, {.label = "unsafe"});
    const std::string svg = figure().svg();
    check(svg.find(CORAcolor("CORA:unsafeLight").hex()) != std::string::npos,
          b + ": the region is drawn in the unsafe color");
    check(svg.find("unsafe") != std::string::npos, b + ": legend entry");
    figure().clear();
    plot(initialSet(), {0, 1});
    plot(Specification::safeSet(test::column({1, 0}), 1.05), {0, 1});
    check(figure().svg().find(CORAcolor("CORA:unsafe").hex()) != std::string::npos,
          b + ": a safe set shades the outside");
}

void trajectories_and_points(const std::string &b) {
    figure().clear();
    const LinearSys sys(Tensor({{-0.1, 1}, {-1, -0.1}}));
    cora::Rng rng(1);
    plot(sys.simulateRandom(initialSet(), 3, 0.1, 1.0, rng), {0, 1}, {.label = "simulations"});
    plot(initialSet().randPoint(5, rng), {0, 1});
    const std::string svg = figure().svg();
    check(count(svg, "<circle") == 5, b + ": only the points are dots, trajectories have no markers");
    check(svg.find("simulations") != std::string::npos, b + ": legend entry");
}

void plot_rejects_bad_dimensions(const std::string &b) {
    figure().clear();
    check(test::throws([] { plot(initialSet(), {0}); }), b + ": one dimension is not a plot");
    check(test::throws([] { plot(initialSet(), {0, 2}); }), b + ": dimension 2 does not exist");
    check(test::throws([] { plot(Tensor({{1, 2}, {3, 4}}), {0, 5}); }), b + ": points too");
}

void save_writes_the_file(const std::string &b) {
    figure().clear();
    plot(initialSet());
    const std::string path = "test_plot_svg_output.svg";
    figure().save(path);
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    check(text.str().rfind("<svg", 0) == 0, b + ": the file is an SVG");
    std::remove(path.c_str());
    check(test::throws([] { figure().save("/nonexistent-folder/figure.svg"); }),
          b + ": an unwritable path is reported");
}

void colors_are_CORA_colors() {
    check(test::close(CORAcolor("CORA:blue").g, 0.4470), "CORA:blue");
    check(test::close(CORAcolor("CORA:color2").r, 0.8500),
          "CORA:color2 is the second palette color");
    check(test::close(CORAcolor("CORA:reachSet", 3, 3).r, 0.2706), "the last reach color is blue");
    check(test::close(CORAcolor("CORA:reachSet", 3, 1).r, 0.6902),
          "the first reach color is light");
    check(test::close(CORAcolor("CORA:red:light").r, 0.8 + 0.8500 * 0.2), "the light variant");
    check(test::throws([] { CORAcolor("CORA:nope"); }), "an unknown color throws");
    check(test::throws([] { CORAcolor("blue"); }), "a color needs the CORA: prefix");
    check(test::throws([] { CORAcolor("CORA:blue:pale"); }), "an unknown variant throws");
    check(test::throws([] { CORAcolor("CORA:reachSet", 2, 3); }), "the index must not exceed");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        a_set_is_an_initial_set_unless_told_otherwise(b);
        a_reach_set_is_drawn_as_one_union(b);
        a_specification_is_shaded_up_to_the_limits(b);
        trajectories_and_points(b);
        plot_rejects_bad_dimensions(b);
        save_writes_the_file(b);
    });
    colors_are_CORA_colors();
    return test::finish("plot");
}
