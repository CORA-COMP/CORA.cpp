// test_zonotope_polytope - Zonotope::polytope: the facets of a zonotope as halfspaces
//
// CORA: contSet/@zonotope/polytope.m
// CORA authors: Niklas Kochdumper, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "cora/global/tensor/tolerance.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The facets of a random zonotope hold exactly its extreme points (up to the facet tolerance) and
/// are 2 C(m, n - 1) at most; a point beyond the support function of a facet is outside.
void facets_bound_the_zonotope(const std::string &b) {
    Rng rng(83);
    for (int n = 1; n <= 4; ++n) {
        const Zonotope Z = Zonotope::generate_random(rng, n);
        const Polytope P = Z.polytope();
        check(P.A().shape()[1] == n && P.A().shape()[0] % 2 == 0,
              b + ": halfspaces in " + std::to_string(n) + "D");
        check(P.contains(Z.rand_point(30, rng, "extreme"), "exact", numerics::kPolytopeFacet) &&
                  P.contains(Z.center()),
              b + ": the zonotope is inside, " + std::to_string(n) + "D");
        const Tensor outside = Z.center() + Z.generators().abs().sum(-1, true) * 1.1 + 1.0;
        check(!P.contains(outside),
              b + ": a point far away is outside, " + std::to_string(n) + "D");
    }
}

/// A square has four facets, a point its box and a segment in the plane two.
void exact_forms(const std::string &b) {
    const Zonotope square(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    const Polytope P = square.polytope();
    check(P.A().shape() == Shape{4, 2} && P.contains(column({2.0, 5.0})) &&
              !P.contains(column({2.1, 5.0})) && P.contains(column({0.0, -1.0})),
          b + ": the square");
    const Zonotope point(column({1.0, 2.0}), Tensor::zeros({2, 0}));
    check(point.polytope().A().shape() == Shape{4, 2}, b + ": the box of a point");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(facets_bound_the_zonotope);
    test::for_each_backend(exact_forms);
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { Zonotope::stack({Z, Z}).polytope(); }, Error::not_supported),
          "polytope: a batch");
    ChecksScope off(false);
    check(Z.polytope().A().shape() == Shape{2, 1}, "polytope: the same with the checks off");
    return test::finish("zonotope polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
