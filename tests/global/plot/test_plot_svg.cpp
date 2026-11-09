// test_plot_svg - plot: sets, reachable sets, specifications and trajectories into a figure, the
//
// color schemes, and the layers a wrapper takes out of it
//
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"
#include "spec_testing.h"
#include "testing.h"

#include <cstdio>
#include <fstream>
#include <sstream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

std::size_t count(const std::string &text, const std::string &what) {
    std::size_t n = 0;
    for (std::size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1))
        ++n;
    return n;
}

bool has(const std::string &text, const std::string &what) {
    return text.find(what) != std::string::npos;
}

// The initial set, made on the backend that is current.
Zonotope initial_set() { return Zonotope(test::column({1, 0}), Tensor({{0.1, 0}, {0, 0.1}})); }

ReachSet oscillator() {
    return LinearSys(Tensor({{-0.1, 1}, {-1, -0.1}}))
        .reach({.initial_set = initial_set(), .t_final = 2.0},
               alg::Standard{.time_step = 0.1, .taylor_terms = 6});
}

// A set takes the next color of CORA's order, filled: blue, red, yellow, ...
void sets_take_the_colors_of_cora_in_turn(const std::string &b) {
    use_cora_colors("CORA:default");
    figure().clear();
    plot(initial_set(), {0, 1}, {.label = "first"});
    check(has(figure().svg(), "fill=\"" + cora_color("CORA:blue").hex() + "\""),
          b + ": the first set is filled blue");
    plot(initial_set());
    plot(initial_set());
    const std::vector<Figure::Layer> &layers = figure().layers();
    check(layers.size() == 3 && layers[1].face.hex() == cora_color("CORA:red").hex() &&
              layers[2].face.hex() == cora_color("CORA:yellow").hex(),
          b + ": then red, then yellow");
    for (int i = 0; i < 6; ++i) plot(initial_set()); // the ninth set
    check(figure().layers().back().face.hex() == cora_color("CORA:red").hex(),
          b + ": the ninth set is red again: the order repeats after seven colors");
    figure().clear();
    plot(initial_set());
    check(figure().layers()[0].face.hex() == cora_color("CORA:blue").hex(),
          b + ": clearing the figure starts again with blue");
}

void a_set_is_filled_unless_told_otherwise(const std::string &b) {
    use_cora_colors("CORA:default");
    figure().clear();
    plot(initial_set(), {0, 1}, {.filled = false});
    check(has(figure().svg(), "fill=\"none\""), b + ": open when not filled");
    figure().clear();
    plot(initial_set(), {0, 1}, {.color = "CORA:red"});
    const Figure::Layer &both = figure().layers()[0];
    check(both.edge.hex() == cora_color("CORA:red").hex() &&
              both.face.hex() == cora_color("CORA:red").hex(),
          b + ": a color is the outline and the fill");
    figure().clear();
    plot(initial_set(), {0, 1}, {.face_color = "CORA:red"});
    // a face color is the fill
    const Figure::Layer &fill = figure().layers()[0];
    check(fill.face.hex() == cora_color("CORA:red").hex() &&
              fill.edge.hex() == cora_color("CORA:simulations").hex(),
          b + ": a facecolor is the fill, with a black outline");
    figure().clear();
    plot(initial_set(), {0, 1}, {.color = "CORA:red"});
    plot(initial_set());
    check(figure().layers()[1].face.hex() == cora_color("CORA:blue").hex(),
          b + ": a given color does not use up a color of the order");
}

// The scheme of reachability analysis: a set is an initial set, a reachable set is blue.
void the_dynamics_scheme_draws_initial_sets_and_blue_reach_sets(const std::string &b) {
    use_cora_colors("CORA:contDynamics");
    figure().clear();
    plot(initial_set(), {0, 1}, {.label = "initial set"});
    const std::string set = figure().svg();
    check(has(set, "fill=\"#ffffff\"") && has(set, "stroke=\"#000000\""),
          b + ": white with a black outline");
    check(has(set, "initial set"), b + ": the legend names the set");
    plot(initial_set());
    check(figure().layers()[1].face.hex() == "#ffffff", b + ": every set is an initial set");
    figure().clear();
    plot(oscillator());
    check(figure().layers()[0].face.hex() == cora_color("CORA:reachSet", 1, 1).hex(),
          b + ": a reachable set is blue");
    use_cora_colors("CORA:default");
}

void a_reach_set_is_drawn_as_one_union(const std::string &b) {
    use_cora_colors("CORA:default");
    figure().clear();
    const ReachSet R = oscillator();
    // one region for all intervals
    plot(R, {0, 1}, {.label = "reachable set"});
    // The core has united the sets: one loop and one path, however many intervals there are.
    check(figure().layers()[0].polygons.size() == 1 && count(figure().svg(), "<path") == 1,
          b + ": one region for all intervals");
    check(figure().equal_axes, b + ": a reach plot has equal axes");
    check(figure().layers()[0].face.hex() == cora_color("CORA:blue").hex(),
          b + ": in the default scheme it takes the next color");
    figure().clear();
    plot(R, {0, 1}, {.unify = false});
    check(figure().layers()[0].polygons.size() == R.time_interval.size(),
          b + ": not unified keeps every set");
}

void reach_options_choose_what_is_drawn(const std::string &b) {
    use_cora_colors("CORA:default");
    const ReachSet R = oscillator();
    figure().clear();
    plot(R, {0, 1}, {.time_points = true});
    check(figure().layers().size() == 2 && figure().layers()[1].dashed,
          b + ": the time points are a second, dotted layer");
    check(has(figure().svg(), "stroke-dasharray"), b + ": drawn dashed");
    figure().clear();
    plot(R);
    // every second set
    const std::size_t all = figure().layers()[0].polygons.size();
    figure().clear();
    plot(R, {0, 1}, {.step = 2});
    check(figure().layers()[0].polygons.size() == (all + 1) / 2, b + ": every second set");
    check(test::throws([&] { plot(R, {0, 1}, {.step = 0}); }), b + ": step 0 is refused");
    figure().clear();
    plot(R, {0, 1}, {.filled = false});
    check(!figure().layers()[0].filled, b + ": a reachable set can be an outline");
}

// Layers are stacked by zorder, whatever the order of the plot calls.
void sets_lie_on_top_of_reachable_sets(const std::string &b) {
    use_cora_colors("CORA:contDynamics");
    figure().clear();
    plot(initial_set());
    plot(oscillator());
    const std::string svg = figure().svg();
    check(svg.find(cora_color("CORA:reachSet").hex()) < svg.find("#ffffff"),
          b + ": the reachable set is drawn first although it was plotted last");
    use_cora_colors("CORA:default");
}

// A filled set is translucent (opacity 0.2, as CORA draws it), its outline is not.
void filled_sets_are_translucent(const std::string &b) {
    use_cora_colors("CORA:default");
    figure().clear();
    plot(initial_set());
    check(figure().layers()[0].face_alpha == 0.2, b + ": a filled set has a fill of opacity 0.2");
    check(has(figure().svg(), "fill-opacity=\"0.2\""), b + ": written as fill-opacity");
    figure().clear();
    plot(initial_set(), {0, 1}, {.face_alpha = 0.5});
    check(figure().layers()[0].face_alpha == 0.5, b + ": face_alpha chooses it");
    figure().clear();
    plot(oscillator(), {0, 1}, {.face_alpha = 0.3});
    check(has(figure().svg(), "fill-opacity=\"0.3\""), b + ": a translucent union is one fill");
    figure().clear();
    plot(oscillator());
    check(figure().layers()[0].face_alpha == 1, b + ": a reachable set is opaque");
    use_cora_colors("CORA:contDynamics");
    figure().clear();
    plot(initial_set());
    check(figure().layers()[0].face_alpha == 1, b + ": an initial set is opaque white");
    // restore the default colors
    use_cora_colors("CORA:default");
}

// The corners of a box are corners: outlines are joined with miters, not rounded.
void outlines_have_sharp_corners(const std::string &b) {
    use_cora_colors("CORA:default");
    figure().clear();
    plot(initial_set());
    plot(oscillator());
    const std::string svg = figure().svg();
    check(has(svg, "stroke-linejoin=\"miter\"") && !has(svg, "stroke-linejoin=\"round\""),
          b + ": mitered corners");
}

void a_specification_is_shaded_up_to_the_limits(const std::string &b) {
    figure().clear();
    plot(initial_set(), {0, 1});
    plot(Specification::unsafe_set(test::halfspace(test::column({1, 0}), 0.5)), {0, 1},
         {.label = "unsafe"});
    const std::string svg = figure().svg();
    check(has(svg, cora_color("CORA:unsafeLight").hex()), b + ": drawn in the unsafe color");
    check(has(svg, "unsafe"), b + ": legend entry");
    figure().clear();
    plot(initial_set(), {0, 1});
    plot(Specification::safe_set(test::halfspace(test::column({1, 0}), 1.05)), {0, 1});
    // a safe set shades the outside
    check(has(figure().svg(), cora_color("CORA:unsafe").hex()),
          b + ": a safe set shades the outside");
}

void trajectories_and_points(const std::string &b) {
    figure().clear();
    const LinearSys sys(Tensor({{-0.1, 1}, {-1, -0.1}}));
    cora::Rng rng(1);
    plot(sys.simulate_random({.initial_set = initial_set(), .t_final = 1.0},
                             alg::Exact{.time_step = 0.1}, 3, rng),
         {0, 1}, {.label = "simulations"});
    plot(initial_set().rand_point(5, rng), {0, 1});
    const std::string svg = figure().svg();
    check(count(svg, "<circle") == 5,
          b + ": only the points are dots, trajectories have no markers");
    check(has(svg, "simulations"), b + ": legend entry");
}

void a_wrapper_takes_the_layers_out(const std::string &b) {
    use_cora_colors("CORA:default");
    figure().clear();
    Figure mine;
    // layers go into the given figure
    plot(mine, initial_set());
    plot(mine, initial_set());
    check(mine.layers().size() == 2, b + ": the layers are in the figure that was given");
    check(figure().layers().empty(), b + ": the current figure is left alone");
    const std::vector<Figure::Layer> taken = mine.take_layers();
    check(taken.size() == 2 && mine.layers().empty(), b + ": take_layers hands them on");
    plot(mine, initial_set());
    check(mine.layers()[0].face.hex() == cora_color("CORA:yellow").hex(),
          b + ": the color counter goes on after taking");
    const Polygon square = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
    const Polygon right = Figure::clip_region(square, {1, 0}, 1, 1);
    check(right.size() == 4 && right[0][0] >= 1 - 1e-12, b + ": clip_region keeps a halfplane");
}

void plot_rejects_bad_dimensions(const std::string &b) {
    figure().clear();
    check(test::throws([] { plot(initial_set(), {0, 1, 2, 3}); }), b + ": four dimensions");
    check(test::throws([] { plot(initial_set(), {}); }), b + ": no dimension");
    check(test::throws([] { plot(initial_set(), {0, 2}); }), b + ": dimension 2 does not exist");
    check(test::throws([] { plot(Tensor({{1, 2}, {3, 4}}), {0, 5}); }), b + ": points too");
}

// save writes the file
void save_writes_the_file(const std::string &b) {
    figure().clear();
    plot(initial_set());
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

void an_unknown_color_scheme_is_described() {
    check(test::throws([] { use_cora_colors("CORA:rainbow"); }), "an unknown scheme throws");
    try {
        use_cora_colors("CORA:rainbow");
    } catch (const Error &e) {
        // the message names the schemes
        check(has(e.what(), "CORA:default") && has(e.what(), "CORA:contDynamics"),
              "the message names the schemes there are");
    }
}

void colors_are_cora_colors() {
    check(test::close(cora_color("CORA:blue").g, 0.4470), "CORA:blue");
    check(test::close(cora_color("CORA:color2").r, 0.8500),
          "CORA:color2 is the second palette color");
    check(test::close(cora_color("CORA:reachSet", 3, 3).r, 0.2706), "the last reach color is blue");
    check(test::close(cora_color("CORA:reachSet", 3, 1).r, 0.6902),
          "the first reach color is light");
    check(test::close(cora_color("CORA:red:light").r, 0.8 + 0.8500 * 0.2), "the light variant");
    check(test::throws([] { cora_color("CORA:nope"); }), "an unknown color throws");
    check(test::throws([] { cora_color("blue"); }), "a color needs the CORA: prefix");
    check(test::throws([] { cora_color("CORA:blue:pale"); }), "an unknown variant throws");
    check(test::throws([] { cora_color("CORA:reachSet", 2, 3); }), "the index must not exceed");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sets_take_the_colors_of_cora_in_turn(b);
        a_set_is_filled_unless_told_otherwise(b);
        the_dynamics_scheme_draws_initial_sets_and_blue_reach_sets(b);
        a_reach_set_is_drawn_as_one_union(b);
        reach_options_choose_what_is_drawn(b);
        sets_lie_on_top_of_reachable_sets(b);
        filled_sets_are_translucent(b);
        outlines_have_sharp_corners(b);
        a_specification_is_shaded_up_to_the_limits(b);
        trajectories_and_points(b);
        a_wrapper_takes_the_layers_out(b);
        plot_rejects_bad_dimensions(b);
        save_writes_the_file(b);
    });
    an_unknown_color_scheme_is_described();
    colors_are_cora_colors();
    return test::finish("plot");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
