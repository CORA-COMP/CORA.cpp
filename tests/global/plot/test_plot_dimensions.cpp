// test_plot_dimensions - plot of one, two and three dimensions, reachable sets over time, the
//
// time-point sets, random points and the options of CORA's plot that the core knows
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

bool has(const std::string &text, const std::string &what) {
    return text.find(what) != std::string::npos;
}

Zonotope box_set() {
    return Zonotope(test::column({1, 0, 2}), Tensor({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}));
}

ReachSet oscillator() {
    return LinearSys(Tensor({{-0.1, 1}, {-1, -0.1}}))
        .reach({.initial_set = Zonotope(test::column({1, 0}), Tensor({{0.1, 0}, {0, 0.1}})),
                .t_final = 1.0},
               alg::Standard{.time_step = 0.25, .taylor_terms = 6});
}

// One dimension is a bar on the x axis from the lower to the upper bound.
void one_dimension_is_a_bar(const std::string &b) {
    use_cora_colors("CORA:default");
    Figure fig;
    plot(fig, box_set(), {1});
    const Figure::Layer &l = fig.layers()[0];
    check(l.kind == Figure::Kind::polygons && l.polygons.size() == 1 && l.polygons[0].size() == 2,
          b + ": one polygon of two points");
    check(close(l.polygons[0][0][0], -2) && close(l.polygons[0][1][0], 2) &&
              l.polygons[0][0][1] == 0,
          b + ": from -2 to 2 at height 0");
    check(fig.xlabel == "x[1]" && fig.ylabel.empty(),
          b + ": the x axis is named, the y axis is not");
    Figure point;
    plot(point, Zonotope(test::column({1, 2}), Tensor::zeros({2, 1})), {0});
    check(point.layers()[0].polygons[0].size() == 1, b + ": a point is one point");
    check(has(fig.svg(), "<path"), b + ": drawn as SVG");
}

// Three dimensions are the faces of the polytope, counter-clockwise, in a three-dimensional figure.
void three_dimensions_are_faces(const std::string &b) {
    use_cora_colors("CORA:default");
    Figure fig;
    plot(fig, box_set(), {0, 1, 2}, {.label = "box"});
    check(fig.dimension == 3 && fig.layers()[0].kind == Figure::Kind::surface, b + ": a surface");
    check(fig.layers()[0].faces.size() == 6, b + ": six faces for a box");
    check(fig.zlabel == "x[2]", b + ": three axes are named");
    const Figure::Limits l = fig.limits();
    check(l.z0 < 2 - 3 && l.z1 > 2 + 3 && l.x0 < 0 && l.y1 > 2, b + ": the limits hold the box");
    check(has(fig.svg(), "<path"), b + ": drawn as SVG");
    check(test::throws([&] { plot(fig, box_set(), {0, 1}); }),
          b + ": 2-D layers need their own figure");
    Figure flat;
    plot(flat, box_set(), {0, 1});
    check(test::throws([&] { plot(flat, box_set(), {0, 1, 2}); }), b + ": and the other way round");
}

// The edge and the face have colors of their own, as CORA's EdgeColor and FaceColor.
void colors_of_edge_and_face(const std::string &b) {
    use_cora_colors("CORA:default");
    Figure fig;
    plot(fig, box_set(), {0, 1}, {.face_color = "CORA:red", .edge_color = "CORA:green"});
    const Figure::Layer &l = fig.layers()[0];
    check(l.face.hex() == cora_color("CORA:red").hex() &&
              l.edge.hex() == cora_color("CORA:green").hex(),
          b + ": a face color and an edge color");
    Figure only_edge;
    plot(only_edge, box_set(), {0, 1}, {.edge_color = "CORA:green"});
    check(only_edge.layers()[0].edge.hex() == cora_color("CORA:green").hex() &&
              only_edge.layers()[0].face.hex() == cora_color("CORA:blue").hex(),
          b + ": an edge color alone keeps the next color as the fill");
}

// A reachable set over time is boxes over the time intervals, united; the time points are lines.
void a_reach_set_over_time(const std::string &b) {
    use_cora_colors("CORA:default");
    const ReachSet R = oscillator();
    check(close(R.time_step, 0.25), b + ": the algorithm records its step");
    Figure fig;
    plot_over_time(fig, R, 0, {.label = "x1"});
    const Figure::Layer &l = fig.layers()[0];
    check(l.polygons.size() == 1 && l.filled && fig.xlabel == "t" && fig.ylabel == "x[0]",
          b + ": one region, with the axes named");
    double t0 = 1e300, t1 = -1e300;
    for (const Point &p : l.polygons[0]) {
        t0 = std::min(t0, p[0]);
        t1 = std::max(t1, p[0]);
    }
    check(close(t0, 0) && close(t1, 1.0), b + ": from 0 to t_final");
    Figure tp;
    plot_over_time(tp, R, 1, {.set = "tp"});
    check(tp.layers()[0].polygons.size() == R.time_point.size() && !tp.layers()[0].filled &&
              tp.layers()[0].polygons[0].size() == 2 && tp.layers()[0].polygons[0][0][0] == 0,
          b + ": time points are vertical lines at their times");
    Figure steps;
    plot_over_time(steps, R, 0, {.unify = false, .step = 2});
    check(steps.layers()[0].polygons.size() == (R.time_interval.size() + 1) / 2,
          b + ": every second set, not united");
    ReachSet unknown = R;
    unknown.time_step = 0;
    check(test::throws([&] { plot_over_time(fig, unknown, 0); }), b + ": without a time step");
    check(test::throws([&] { plot_over_time(fig, R, 5); }), b + ": a dimension that is not there");
}

void simulations_over_time(const std::string &b) {
    const LinearSys sys(Tensor({{-0.1, 1}, {-1, -0.1}}));
    cora::Rng rng(2);
    const Zonotope X0(test::column({1, 0}), Tensor({{0.1, 0}, {0, 0.1}}));
    const std::vector<Tensor> x = sys.simulate_random({.initial_set = X0, .t_final = 1.0},
                                                      alg::Exact{.time_step = 0.25}, 3, rng);
    Figure fig;
    plot_over_time(fig, x, 1, 0.25, {.label = "simulations"});
    check(fig.layers().size() == 3 && fig.layers()[0].kind == Figure::Kind::polyline,
          b + ": a line per trajectory");
    check(close(fig.layers()[0].polygons[0].back()[0], 1.0), b + ": the last point is at t_final");
    check(test::throws([&] { plot_over_time(fig, x, 1, 0.0); }), b + ": a time step is positive");
    check(test::throws([&] { plot_over_time(fig, {}, 1, 0.25); }), b + ": no points");
}

// CORA's `Set`: the time-point sets instead of the time-interval sets.
void time_point_sets_instead(const std::string &b) {
    use_cora_colors("CORA:default");
    const ReachSet R = oscillator();
    Figure fig;
    plot(fig, R, {0, 1}, {.set = "tp"});
    check(fig.layers()[0].polygons.size() <= R.time_point.size(), b + ": the time-point sets");
    check(test::throws([&] { plot(fig, R, {0, 1}, {.set = "xx"}); }), b + ": an unknown set");
    Figure three;
    plot(three, R, {0, 1, 0}); // a flat set in space: its faces are polygons
    check(three.dimension == 3 && three.layers()[0].kind == Figure::Kind::surface,
          b + ": a reachable set also plots three dimensions");
}

void random_points_of_a_set(const std::string &b) {
    cora::Rng rng(3);
    Figure fig;
    plot_rand_point(fig, box_set(), {0, 2}, 10, rng, {.label = "points"});
    check(fig.layers().size() == 1 && fig.layers()[0].kind == Figure::Kind::points &&
              fig.layers()[0].polygons[0].size() == 10,
          b + ": ten dots");
    for (const Point &p : fig.layers()[0].polygons[0])
        check(p[0] >= -0.001 && p[0] <= 2.001 && p[1] >= -1.001 && p[1] <= 5.001,
              b + ": inside the set");
    ChecksScope off(false);
    plot_rand_point(fig, box_set(), {0, 2}, 3, rng);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        one_dimension_is_a_bar(b);
        three_dimensions_are_faces(b);
        colors_of_edge_and_face(b);
        a_reach_set_over_time(b);
        simulations_over_time(b);
        time_point_sets_instead(b);
        random_points_of_a_set(b);
    });
    return test::finish("plot dimensions");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
