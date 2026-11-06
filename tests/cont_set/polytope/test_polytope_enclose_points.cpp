// test_polytope_enclose_points - Polytope::enclose_points: the hull of a cloud of points
//
// CORA: contSet/@polytope/enclosePoints.m
// PROPERTY: contSet/@polytope/enclosePoints.m
//
// Property: the vertices are the extreme points of the cloud, and every point is inside
//
// CORA authors: Niklas Kochdumper, Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

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
    // a sphere and its inside
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

/// Every point of the cloud is in the hull and the vertices are points of the cloud; points that
/// are convex combinations of the others are dropped.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3, 4}) {
        const Tensor sphere = test::sphere_points(rng, n, n + 4);
        const Tensor weights = Tensor::ones({n + 4, 3}) * (1.0 / (n + 4));
        const Tensor inner = sphere.matmul(weights);
        const Polytope P = Polytope::enclose_points(Tensor::concat({sphere, inner}, -1));
        check(test::same_columns(P.V(), sphere),
              b + ": only the sphere stays, " + std::to_string(n) + "D");
        check(P.contains(inner, "exact", 1e-8),
              b + ": the inner points are inside, " + std::to_string(n) + "D");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        clouds(b);
        properties(b);
    });
    refusals();
    return test::finish("polytope enclose_points");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
