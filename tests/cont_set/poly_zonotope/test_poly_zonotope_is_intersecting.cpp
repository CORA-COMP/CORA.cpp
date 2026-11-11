// test_poly_zonotope_is_intersecting - PolyZonotope is_intersecting: a polytope may be met
//
// CORA: contSet/@polyZonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/registry.h"
#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// pZ = {(1, 0) + a (1, 1) + a b (0, 2) + (0, 1) t}: generators on a and a b, one independent.
PolyZonotope known() {
    return {column({1.0, 0.0}),
            Tensor::from_data({1, 0, 1, 2}, {2, 2}),
            Tensor::from_data({0, 1}, {2, 1}),
            {{1, 0}, {1, 1}},
            {1, 2}};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const PolyZonotope P = known();
        // x1 in [0, 2]: the halfspace x1 <= -1 is missed, x1 <= 1 is met
        const Polytope miss(Tensor::from_data({1.0, 0.0}, {1, 2}), column({-1.0}));
        const Polytope hit(Tensor::from_data({1.0, 0.0}, {1, 2}), column({1.0}));
        check(!P.is_intersecting(miss) && P.is_intersecting(hit), b + ": a halfspace");
        const Polytope plane(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}),
                             Tensor::from_data({1.0, 0.0}, {1, 2}), column({5.0}));
        check(!P.is_intersecting(plane), b + ": a hyperplane outside the range");
        check(is_intersecting(P, miss) == false, b + ": through the registry");
    });
    check(test::raises(
              [] { known().is_intersecting(Polytope(Tensor::zeros({1, 3}), column({0.0}))); },
              Error::dimension_mismatch),
          "is_intersecting: dimension");
    ChecksScope off(false);
    check(!known().is_intersecting(Polytope(Tensor::from_data({1.0, 0.0}, {1, 2}), column({-1.0}))),
          "is_intersecting: valid with the checks off");
    return test::finish("poly_zonotope is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
