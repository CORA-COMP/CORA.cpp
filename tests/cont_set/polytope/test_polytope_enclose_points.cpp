// test_polytope_enclose_points - Polytope::enclose_points: the hull of a cloud of points
//
// CORA: contSet/@polytope/enclosePoints.m
// PROPERTY: the vertices are the extreme points of the cloud, and every point is inside

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Extreme points are kept in order, the others go; degenerate clouds work.
void clouds(const std::string &b) {
    const Polytope P =
        Polytope::enclose_points(Tensor({{0.0, 1.0, 1.0, 0.0, 0.5}, {0.0, 0.0, 1.0, 1.0, 0.5}}));
    check(P.is_v_rep() && close(P.V(), {0, 1, 1, 0, 0, 0, 1, 1}, 0), b + ": the inner point goes");
    check(close(Polytope::enclose_points(Tensor({{1.0, 1.0}, {2.0, 2.0}})).V(), {1, 2}, 0),
          b + ": equal points: the first");
    check(Polytope::enclose_points(Tensor({{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}})).V().shape() ==
              Shape{2, 2},
          b + ": collinear points: the ends");
    check(Polytope::enclose_points(Tensor::zeros({3, 4})).V().shape() == Shape{3, 1},
          b + ": all at the origin");
    check(Polytope::enclose_points(
              Tensor({{0.0, 1.0, 0.0, 1.0}, {0.0, 0.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}}))
                  .V()
                  .shape() == Shape{3, 4},
          b + ": a flat square in space");
    check(close(Polytope::enclose_points(Tensor({{3.0, 1.0, 2.0}})).V(), {1, 3}, 0), b + ": 1D");
    Rng rng(5);
    const Tensor sphere = test::sphere_points(rng, 3, 8);
    const Tensor mean = sphere.mean(-1, true);
    const Tensor both = Tensor::concat({sphere, mean + (sphere - mean) * 0.3}, -1);
    check(test::same_columns(Polytope::enclose_points(both).V(), sphere),
          b + ": a sphere and its inside");
    check(Polytope::enclose_points(both).contains(both, "exact", 1e-8),
          b + ": every point is inside");
}

/// What is refused; checks off.
void refusals() {
    check(test::raises([&] { Polytope::enclose_points(Tensor::zeros({3})); }, Error::wrong_value),
          "enclose_points: a matrix");
    check(test::raises([&] { Polytope::enclose_points(Tensor::ones({2, 2, 3})); },
                       Error::wrong_value),
          "enclose_points: not a batch");
    ChecksScope off(false);
    check(Polytope::enclose_points(Tensor({{0.0, 1.0}, {0.0, 1.0}})).V().shape() == Shape{2, 2},
          "enclose_points: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { clouds(b); });
    refusals();
    return test::finish("polytope enclose_points");
}
