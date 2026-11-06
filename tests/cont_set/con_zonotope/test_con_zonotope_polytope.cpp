// test_con_zonotope_polytope - ConZonotope::polytope: the halfspace form of a constrained zonotope
//
// CORA: contSet/@conZonotope/polytope.m
// PROPERTY: contSet/@conZonotope/polytope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
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

/// Hand-computed: the hexagon contains its corners and midpoints, not the points beyond a facet.
void polytope_known(const std::string &b) {
    const Polytope P = test::hexagon().polytope();
    check(P.contains(column({1.0, 0.0})) && P.contains(column({-1.0, 1.0})) &&
              P.contains(column({0.5, -0.5})) && P.contains(column({0.0, 0.0})),
          b + ": points of the hexagon");
    check(!P.contains(column({1.0, 1.0})) && !P.contains(column({-1.0, -1.0})) &&
              !P.contains(column({1.1, 0.0})),
          b + ": points outside");
}

/// Stated property: the sampled points are in the polytope.
void polytope_property(const std::string &b) {
    Rng rng(76);
    const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, 2);
    check(cz.polytope().contains(test::sample_points(cz, rng, 20)), b + ": the points are inside");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        polytope_known(b);
        polytope_property(b);
    });
    const Tensor batched = Tensor::from_data({0.0, 0.0, 1.0, 1.0}, {2, 2, 1});
    check(test::raises(
              [&] {
                  ConZonotope(batched, Tensor::from_data({1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0},
                                                         {2, 2, 2}))
                      .polytope();
              },
              Error::not_supported),
          "polytope: a batch");
    ChecksScope off(false);
    check(test::hexagon().polytope().dim() == 2, "polytope: the same with the checks off");
    return test::finish("con_zonotope polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
