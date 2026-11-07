// test_capsule_polytope - Capsule polytope: exact for a segment, outer for the rest
//
// CORA: contSet/@capsule/polytope.m
// PROPERTY: contSet/@capsule/polytope.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/rng.h"

#include <cmath>
#include <numbers>

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: one dimension, a point, a segment and the empty capsule.
void polytope_known(const std::string &b) {
    const Capsule line(column({1.0}), column({2.0}), 0.5);
    const Polytope P1 = line.polytope();
    check(P1.contains(column({-1.4})) && P1.contains(column({3.4})) &&
              !P1.contains(column({3.6})) && !P1.contains(column({-1.6})),
          b + ": 1D is [-1.5, 3.5]");
    check(Capsule(column({1.0, 2.0})).polytope().contains(column({1.0, 2.0})), b + ": a point");
    const Capsule segment(column({1.0, 1.0}), column({1.0, 1.0}), 0.0);
    const Polytope S = segment.polytope();
    check(S.contains(column({1.0, 1.0})) && S.contains(column({1.9, 1.9})) &&
              !S.contains(column({1.0, 1.5})) && !S.contains(column({2.2, 2.2})),
          b + ": a segment");
    check(!Capsule::empty(2).polytope().contains(column({0.0, 0.0})), b + ": empty");
}

/// Stated property: the outer polytope of a capsule holds its points, by method "outer", also for a
/// ball.
void polytope_property(const std::string &b) {
    Rng rng(71);
    for (int k = 0; k < 4; ++k) {
        const Capsule C = k == 3 ? Capsule(column({0.5, 1.0, 2.0}), column({0.0, 0.0, 0.0}), 0.7)
                                 : test::random_capsule(rng, 3);
        check(C.polytope("outer").contains(test::sample_points(C, rng, 20), "exact", 1e-9),
              b + ": the points are inside");
    }
}

/// Exact has no polytope for a proper capsule, inner is not supported, a bad method, a batch.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(test::raises([&] { C.polytope(); }, Error::no_exact_alg), "polytope: exact");
    check(test::raises([&] { C.polytope("inner"); }, Error::not_supported), "polytope: inner");
    check(test::raises([&] { C.polytope("middle"); }, Error::wrong_value), "polytope: method");
    ChecksScope off(false);
    check(C.polytope("outer").dim() == 2, "polytope: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        polytope_known(b);
        polytope_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
