// test_polytope_rescale_cols - Polytope::rescale_cols: the rows in the coordinates of a box
//
// The square in the coordinates of the box [0, 1]^2 (center 0.5, radius 0.5) is the unit box;
// equalities and a batch shift and scale alike. The bits are those of the factor rows of
// ConZonotope::rescale, which the oracle tests compare. The stated property: y is in
// P.rescale_cols(mid, rad) exactly if mid + diag(rad) y is in P.
//
// CORA.cpp only: tests CORALean's HPolytope.rescaleCols
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The square in the coordinates of its box, equalities, and a batch.
void rows(const std::string &b) {
    const Polytope P = test::square_h().rescale_cols(vec({0.5, 0.5}), vec({0.5, 0.5}));
    check(close(P.A(), {0.5, 0, 0, 0.5, -0.5, 0, 0, -0.5}, 0.0) &&
              close(P.b(), {0.5, 0.5, 0.5, 0.5}, 0.0),
          b + ": the square is the unit box");
    const Polytope E = Polytope::equalities(Tensor({{1.0, 2.0}}), vec({3.0}));
    const Polytope R = E.rescale_cols(vec({1.0, 1.0}), vec({2.0, 0.5}));
    check(close(R.Ae(), {2, 1}, 0.0) && close(R.be(), {0}, 0.0), b + ": equalities");
    const Polytope B(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1}));
    const Polytope RB = B.rescale_cols(Tensor({{1.0, 0.0}, {0.0, 0.0}}), vec({1.0, 1.0}));
    check(close(RB.b(), {0, 1}, 0.0), b + ": a batch, a center per member");
}

/// y in P.rescale_cols(mid, rad) exactly if mid + diag(rad) y in P.
void properties(const std::string &b) {
    Rng rng(13);
    const Polytope P = test::random_polytope(rng, 2, 4);
    const Polytope S = P.rescale_cols(vec({0.25, -0.5}), vec({1.5, 0.75}));
    bool same = true;
    for (int k = 0; k < 30; ++k) {
        const std::vector<double> y = test::random_direction(rng, 2);
        const Tensor x = vec({0.25 + 1.5 * y[0], -0.5 + 0.75 * y[1]});
        same = same && S.contains(vec(y)) == P.contains(x);
    }
    check(same, b + ": the coordinates of the box");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().rescale_cols(vec({0.0}), vec({1.0, 1.0})); },
                       Error::dimension_mismatch),
          "rescale_cols: a center per coordinate");
    ChecksScope off(false);
    check(test::square_h().rescale_cols(vec({0.0, 0.0}), vec({1.0, 1.0})).dim() == 2,
          "rescale_cols: the same with the checks off");
    return test::finish("polytope rescale_cols");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
