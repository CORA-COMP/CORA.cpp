// test_con_zonotope_intersect - ConZonotope::intersect: the exact intersection with a set
//
// CORA: contSet/@conZonotope/and_.m
// PROPERTY: contSet/@conZonotope/and_.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"
#include "cora/cont_set/registry.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: [0, 2] and [1, 3] meet in [1, 2], through the agreement row; a zonotope and a
/// box.
void intersect_known(const std::string &b) {
    const ConZonotope A(column({1.0}), Tensor({{1.0}}));
    const ConZonotope B(column({2.0}), Tensor({{1.0}}));
    const ConZonotope R = A.intersect(B);
    check(close(R.c(), {1}, 0) && close(R.generators(), {1, 0}, 0),
          b + ": the first set's generators");
    check(close(R.A(), {1, -1}, 0) && close(R.b(), {1}, 0),
          b + ": one agreement row c + G x = c2 + G2 x2");
    check(close(R.interval().inf(), {1}, 1e-9) && close(R.interval().sup(), {2}, 1e-9),
          b + ": the intersection [1, 2]");
    // the constraints of both sets stay; zero columns and rows go
    const ConZonotope C(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 1.0}}),
                        column({1.0}));
    const ConZonotope D = C.intersect(Zonotope(column({0.5, 0.5}), Tensor({{0.0}, {0.0}})));
    check(close(D.interval().inf(), {0.5, 0.5}, 1e-9), b + ": a point of the segment");
    const ConZonotope E = A.intersect(Interval(column({1.5}), column({4.0})));
    check(close(E.interval().inf(), {1.5}, 1e-9) && close(E.interval().sup(), {2}, 1e-9),
          b + ": a box");
    check(A.intersect(Interval(column({5.0}), column({6.0}))).interval().is_empty(),
          b + ": disjoint sets");
    const Set via_set = intersect(A, Set(B));
    check(via_set.is<ConZonotope>(), b + ": through the registry");
}

/// Stated property: a point is in the intersection exactly if it is in both sets.
void intersect_property(const std::string &b) {
    cora::Rng rng(26);
    for (int k = 0; k < 4; ++k) {
        const ConZonotope A = test::random_con_zonotope(rng, 2, 4, 1);
        const ConZonotope B = test::random_con_zonotope(rng, 2, 3, 1);
        const ConZonotope R = A.intersect(B);
        const Tensor points = test::sample_points(A, rng, 60);
        for (int64_t j = 0; j < points.shape()[1]; ++j) {
            const Tensor p = points.slice(1, j, j + 1);
            const bool in_both = B.contains(p, "exact", 1e-7);
            check(R.contains(p, "exact", 1e-7) == in_both, b + ": in the intersection iff in both");
        }
    }
}

/// Wrong operands are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const ConZonotope A(column({1.0}), Tensor({{1.0}}));
    check(
        test::raises([&] { A.intersect(ConZonotope(column({1.0, 2.0}), Tensor({{1.0}, {1.0}}))); },
                     Error::dimension_mismatch),
        "intersect: different dimensions");
    const Tensor on = A.intersect(A).A();
    ChecksScope off(false);
    check(close(A.intersect(A).A(), on, 0), "intersect: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        intersect_known(b);
        intersect_property(b);
    });
    checks_and_their_switch();
    return test::finish("con_zonotope intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
