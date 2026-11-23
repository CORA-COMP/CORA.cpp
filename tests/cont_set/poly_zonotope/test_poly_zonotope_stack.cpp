// test_poly_zonotope_stack - PolyZonotope stack: sets one below the other with shared factors
//
// The stacked set has the points (x, y) of the two sets at the same factors; sets without
// dependent generators are added to the center, as are three or four sets in pairs.
//
// CORA: contSet/@polyZonotope/stack.m
// PROPERTY: contSet/@polyZonotope/stack.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(23);
        const PolyZonotope P = test::random_poly(rng, 2), Q = test::random_poly(rng, 1);
        const PolyZonotope S = PolyZonotope::stack({P, Q});
        check(S.dim() == 3, b + ": the dimensions add");
        const std::map<int64_t, double> alpha = {{1, 0.4}, {2, -0.8}};
        const std::vector<double> sp = test::point(S, alpha, {0.2, 0.3, -0.4, 0.5}),
                                  pp = test::point(P, alpha, {0.2, 0.3}),
                                  qp = test::point(Q, alpha, {-0.4, 0.5});
        check(
            close(Tensor::from_data(sp, {3}), Tensor::from_data({pp[0], pp[1], qp[0]}, {3}), 1e-12),
            b + ": a point is the pair of the points at equal factors");
        // a point (no generators) joins the center
        const PolyZonotope V(vec({7.0}), Tensor::zeros({0, 1}), Tensor::zeros({0, 1}), {}, {});
        const PolyZonotope W = PolyZonotope::stack({P, V});
        check(W.dim() == 3 && close(W.center().slice(-1, 2, 3), {7.0}) &&
                  W.num_dependent() == P.num_dependent(),
              b + ": a set without generators");
        check(PolyZonotope::stack({P, Q, V}).dim() == 4 &&
                  PolyZonotope::stack({P, Q, V, V}).dim() == 5,
              b + ": three and four sets");
        check(PolyZonotope::stack({P}).isequal(P), b + ": one set");
    });
    check(test::raises([] { PolyZonotope::stack({}); }, Error::wrong_value), "stack: no set");
    ChecksScope off(false);
    check(PolyZonotope::stack({test::sample_poly()}).dim() == 2, "stack: valid, checks off");
    return test::finish("poly_zonotope stack");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
