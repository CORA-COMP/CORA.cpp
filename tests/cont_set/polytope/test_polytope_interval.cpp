// test_polytope_interval - Polytope::interval: the tightest box around the set
//
// CORA: contSet/@polytope/interval.m
// PROPERTY: contSet/@polytope/interval.m
//
// Property: the box of vertices is their coordinate-wise extremes; of halfspaces it is the 2n
//           linear programs, whose bounds are met by points of the set
//
// CORA authors: Matthias Althoff, Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Vertices: the extremes; a batch works; the empty set is the empty box.
void vertices(const std::string &b) {
    const Interval I = Polytope(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}})).interval();
    check(close(I.inf(), {0, 0}, 0) && close(I.sup(), {1, 1}, 0), b + ": the square");
    check(Polytope::empty(2).interval().is_empty(), b + ": the empty set");
    const Interval batch = Polytope(Tensor::ones({3, 2, 4})).interval();
    check(batch.inf().shape() == Shape{3, 2, 1}, b + ": a batch of three");
}

/// Halfspaces: the linear programs; unbounded sides are infinite; an infeasible set is empty.
void halfspaces(const std::string &b) {
    const Polytope tri(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), column({0.0, 0.0, 2.0}));
    const Interval I = tri.interval();
    check(close(I.inf(), {0, 0}, 1e-9) && close(I.sup(), {2, 2}, 1e-9), b + ": a triangle");
    const double inf = std::numeric_limits<double>::infinity();
    const Interval half = Polytope(Tensor({{1.0, 0.0}}), column({1.0})).interval();
    check(half.inf().data()[0] == -inf && half.sup().data()[0] == 1.0 &&
              half.sup().data()[1] == inf,
          b + ": a halfspace: unbounded sides");
    const Polytope none(Tensor({{1.0, 0.0}, {-1.0, 0.0}}), column({0.0, -1.0}));
    check(none.interval().is_empty(), b + ": infeasible: the empty box");
    check(Polytope::inf(2).interval().sup().min().item() == inf, b + ": R^2");
    const Polytope eq(Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
                      column({2.0, 2.0, 2.0, 2.0}), Tensor({{1.0, 1.0}}), column({1.0}));
    check(close(eq.interval().inf(), {-1, -1}, 1e-9) && close(eq.interval().sup(), {2, 2}, 1e-9),
          b + ": with an equality");
}

/// A checks-off call on a valid set.
void refusals() {
    const Polytope P(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    ChecksScope off(false);
    check(close(P.interval().sup(), {1, 1}, 0), "interval: the same with the checks off");
    check(
        test::raises([&] { Polytope(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 1})).interval(); },
                     Error::not_supported),
        "interval: a batch of halfspace polytopes");
}

/// A halfspace polytope built from random vertices has the box of those vertices; the samples of a
/// random halfspace polytope lie in its box.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3, 4}) {
        const Tensor points = test::sphere_points(rng, n, n + 4);
        const Interval box = Polytope(points).interval();
        check(close(box.inf(), points.min(-1, true), 0) &&
                  close(box.sup(), points.max(-1, true), 0),
              b + ": vertices: the extremes, " + std::to_string(n) + "D");
        const Polytope::Constraints C = Polytope(points).constraints();
        const Interval from_rows = Polytope(C.A, C.b).interval();
        check(close(from_rows.inf(), box.inf(), 1e-7) && close(from_rows.sup(), box.sup(), 1e-7),
              b + ": the programs of the rows give the box of the vertices, " + std::to_string(n) +
                  "D");
        const Polytope Q = test::random_polytope(rng, n, 2 * n);
        const Interval hull = Q.interval();
        const Tensor samples = test::sample_points(Q, rng, 30);
        // samples lie in the box
        check((samples - hull.inf()).min().item() >= -1e-9 &&
                  (hull.sup() - samples).min().item() >= -1e-9,
              b + ": samples lie in the box, " + std::to_string(n) + "D");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        properties(b);
        vertices(b);
        halfspaces(b);
    });
    refusals();
    return test::finish("polytope interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
