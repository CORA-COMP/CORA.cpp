// test_capsule_zonotope - Capsule zonotope: the segment plus the box of the ball
//
// CORA: contSet/@capsule/zonotope.m
// PROPERTY: contSet/@capsule/zonotope.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "capsule_testing.h"
#include "cora/cont_set/zonotope/zonotope.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the generators are g and r I; no radius adds none.
void zonotope_known(const std::string &b) {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    const Zonotope Z = C.zonotope();
    // layout: #163 step 2
    check(close(detail::cora_center(Z), {1, 2}, 0) &&
              detail::cora_generators(Z).shape() == Shape({2, 3}),
          b + ": c and 1 + n generators");
    // layout: #163 step 2
    check(close(detail::cora_generators(Z), {3, 0.5, 0, 4, 0, 0.5}, 0), b + ": g, r I");
    check(C.plus(column({0.0, 0.0})).zonotope().dim() == 2, b + ": after plus");
    // layout: #163 step 2
    check(detail::cora_generators(Capsule(column({1.0, 2.0}), column({3.0, 4.0}), 0.0).zonotope())
                  .shape() == Shape({2, 1}),
          b + ": r = 0 adds nothing");
}

/// Stated property: every sampled point of the capsule lies in the zonotope.
void zonotope_property(const std::string &b) {
    Rng rng(67);
    for (int k = 0; k < 3; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        check(C.zonotope().contains(test::sample_points(C, rng, 12), "exact", 1e-9),
              b + ": the points are inside");
    }
}

/// The empty capsule is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    check(test::raises([] { Capsule::empty(2).zonotope(); }, Error::empty_set), "zonotope: empty");
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    ChecksScope off(false);
    // layout: #163 step 2
    check(detail::cora_generators(C.zonotope()).shape() == Shape({2, 3}),
          "zonotope: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        zonotope_known(b);
        zonotope_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
