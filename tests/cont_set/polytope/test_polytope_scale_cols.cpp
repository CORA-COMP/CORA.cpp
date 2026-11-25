// test_polytope_scale_cols - Polytope::scale_cols: the rows in scaled coordinates
//
// The columns of the square and of equalities times the scales, the offsets as they were, and a
// batch with its own scales. The stated property: y is in P.scale_cols(d) exactly if diag(d) y
// is in P.
//
// CORA.cpp only: tests the scaling step of CORALean's HPolytope.rescaleCols
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

/// The columns of the square, of equalities and of a batch scaled.
void rows(const std::string &b) {
    const Polytope P = test::square_h().scale_cols(vec({2.0, -0.5}));
    check(close(P.A(), {2, 0, 0, -0.5, -2, 0, 0, 0.5}, 0.0) && close(P.b(), {1, 1, 0, 0}, 0.0),
          b + ": the square");
    const Polytope E = Polytope::equalities(Tensor({{1.0, 2.0}}), vec({3.0}));
    check(close(E.scale_cols(vec({3.0, 0.25})).Ae(), {3, 0.5}, 0.0), b + ": equalities");
    const Polytope B(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1}));
    check(close(B.scale_cols(Tensor({{1.0, 2.0}, {3.0, 4.0}})).A(), {1, 2, 3, 4}, 0.0),
          b + ": a batch, a scale per member");
}

/// y in P.scale_cols(d) exactly if diag(d) y in P.
void properties(const std::string &b) {
    Rng rng(12);
    const Polytope P = test::random_polytope(rng, 2, 4), S = P.scale_cols(vec({1.5, -0.7}));
    bool same = true;
    for (int k = 0; k < 30; ++k) {
        const std::vector<double> y = test::random_direction(rng, 2);
        same = same && S.contains(vec(y)) == P.contains(vec({1.5 * y[0], -0.7 * y[1]}));
    }
    check(same, b + ": the scaled coordinates");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().scale_cols(vec({1.0, 2.0, 3.0})); },
                       Error::dimension_mismatch),
          "scale_cols: a scale per coordinate");
    ChecksScope off(false);
    check(close(test::square_h().scale_cols(vec({2.0, 2.0})).b(), {1, 1, 0, 0}, 0.0),
          "scale_cols: the same with the checks off");
    return test::finish("polytope scale_cols");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
