// test_poly_zonotope_representsa - PolyZonotope representsa: whether the set is of another type
//
// A point, a box, a zonotope and a set with a squared factor against the types; the types that
// every set is or none is; the tests that are not ported are refused.
//
// Oracle debt: the request is in docs/oracle-requests.md (CORALean has the float definition).
//
// CORA: contSet/@polyZonotope/representsa_.m
// PROPERTY: contSet/@polyZonotope/representsa_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // a random set is a polynomial zonotope, no point and not empty
        Rng rng(4);
        const PolyZonotope R = test::random_poly(rng, 2);
        check(R.representsa("polyZonotope") && !R.representsa("point") &&
                  !R.representsa("emptySet"),
              b + ": a random set");
        const Tensor I2 = Tensor::from_data({1, 0, 0, 1}, {2, 2});
        const PolyZonotope point(vec({1.0, 2.0}), Tensor::zeros({0, 2}), Tensor::zeros({0, 2}), {},
                                 {});
        check(point.representsa("point") && point.representsa("capsule") &&
                  point.representsa("ellipsoid") && !point.representsa("origin"),
              b + ": a point");
        check(PolyZonotope::origin(2).representsa("origin"), b + ": the origin");
        const PolyZonotope box(vec({0.0, 0.0}), I2, Tensor::zeros({0, 2}), {{1, 0}, {0, 1}},
                               {1, 2});
        check(box.representsa("zonotope") && box.representsa("interval") &&
                  box.representsa("polytope") && !box.representsa("parallelotope") &&
                  !box.representsa("point"),
              b + ": a box of two factors (generators dependent: no parallelotope test)");
        // 1D sets are everything
        const PolyZonotope line(vec({0.0}), Tensor::from_data({1, 2}, {2, 1}),
                                Tensor::zeros({0, 1}), {{1}, {2}}, {1});
        check(line.representsa("zonotope") && line.representsa("interval"), b + ": dimension 1");
        // (a, a^2) is no zonotope
        const PolyZonotope parabola(vec({0.0, 0.0}), I2, Tensor::zeros({0, 2}), {{1}, {2}}, {1});
        // the zonotope test needs one factor per generator
        check(!parabola.representsa("zonotope") && !parabola.representsa("interval"),
              b + ": the parabola arc");
        check(box.representsa("polyZonotope") && box.representsa("conPolyZono") &&
                  !box.representsa("emptySet") && !box.representsa("fullspace") &&
                  !box.representsa("halfspace") && !box.representsa("probZonotope") &&
                  !box.representsa("hyperplane"),
              b + ": the types that all or none are");
        const PolyZonotope indep(vec({0.0, 0.0}), Tensor::zeros({0, 2}), I2, {}, {});
        check(indep.representsa("parallelotope"), b + ": independent generators of rank n");
        check(test::raises([&] { parabola.representsa("polytope"); }, Error::not_supported),
              b + ": the polytope test is not ported");
    });
    const PolyZonotope P = test::sample_poly();
    check(test::raises([&] { P.representsa("banana"); }, Error::wrong_value), "representsa: type");
    check(test::raises([&] { P.representsa("polygon"); }, Error::not_supported),
          "representsa: polygon");
    ChecksScope off(false);
    check(P.representsa("polyZonotope"), "representsa: valid with the checks off");
    return test::finish("poly_zonotope representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
