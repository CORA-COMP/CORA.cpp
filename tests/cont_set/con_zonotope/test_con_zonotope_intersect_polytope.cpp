// test_con_zonotope_intersect_polytope - ConZonotope::intersect with a polytope
//
// A square cut by a halfspace and by a hyperplane with an inequality, by hand; on random sets a
// point is in the result exactly if it is in both (the stated property).
//
// CORA: contSet/@polytope/and_.m
// PROPERTY: contSet/@polytope/and_.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/polytope/polytope.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// [-1, 1]^2 cut by x1 <= 0.5 is [-1, 0.5] x [-1, 1]; by x2 = 0.5, x1 <= 0 the segment.
void by_hand(const std::string &b) {
    const ConZonotope square(Zonotope(column({0.0, 0.0}), Tensor::eye(2)));
    const ConZonotope cut = square.intersect(Polytope(Tensor({{1.0, 0.0}}), column({0.5})));
    const Interval I = cut.interval();
    check(close(I.inf(), {-1, -1}, 1e-9) && close(I.sup(), {0.5, 1}, 1e-9), b + ": a halfspace");
    const Polytope line(Tensor({{1.0, 0.0}}), column({0.0}), Tensor({{0.0, 1.0}}), column({0.5}));
    const Interval J = square.intersect(line).interval();
    check(close(J.inf(), {-1, 0.5}, 1e-9) && close(J.sup(), {0, 0.5}, 1e-9),
          b + ": a hyperplane with an inequality");
    const Polytope far(Tensor({{1.0, 0.0}}), column({-3.0}));
    check(square.intersect(far).interval().is_empty(), b + ": disjoint sets");
}

/// Stated property: a point is in the intersection exactly if it is in both sets.
void property(const std::string &b) {
    cora::Rng rng(7);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 4, 1);
        const Polytope P(Tensor({{1.0, 0.5}, {-0.3, 1.0}}), A.c() + column({0.2, 0.1}));
        const ConZonotope R = A.intersect(P);
        const Tensor points = test::sample_points(A, rng, 60);
        for (int64_t j = 0; j < points.shape()[1]; ++j) {
            const Tensor p = points.slice(1, j, j + 1);
            const bool in_both = P.contains(p, "exact", 1e-7);
            check(R.contains(p, "exact", 1e-7) == in_both, b + ": in the intersection iff in both");
        }
    }
}

/// Another dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const ConZonotope A(column({1.0}), Tensor({{1.0}}));
    check(test::raises([&] { A.intersect(Polytope(Tensor({{1.0, 0.0}}), column({0.0}))); },
                       Error::dimension_mismatch),
          "intersect: different dimensions");
    const Polytope P(Tensor({{1.0}}), column({1.5}));
    const Tensor on = A.intersect(P).A();
    ChecksScope off(false);
    check(close(A.intersect(P).A(), on, 0), "intersect: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        by_hand(b);
        property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope intersect polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
