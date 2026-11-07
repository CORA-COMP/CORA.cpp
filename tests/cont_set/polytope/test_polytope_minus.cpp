// test_polytope_minus - Polytope::minus: the polytope shifted by a vector
//
// CORA: contSet/@polytope/minus.m
// PROPERTY: contSet/@polytope/minus.m
//
// Property: x - v is in the shifted set exactly if x is in the set
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Shifts in both forms.
void shifts(const std::string &b) {
    const Tensor v = column({1.0, 2.0});
    check(close(test::square_h().minus(v).b(), {0.0, -1.0, 1.0, 2.0}), b + ": halfspaces");
    check(close(test::square_v().minus(v).V(), {-1.0, 0.0, 0.0, -1.0, -2.0, -2.0, -1.0, -1.0}),
          b + ": vertices");
    check(close((test::square_h() - v).b(), {0.0, -1.0, 1.0, 2.0}), b + ": the operator");
    // the generic ContSet entry does the same
    check(close(static_cast<const ContSet &>(test::square_h()).minus(v)->center(), {-0.5, -1.5},
                1e-8),
          b + ": through ContSet");
}

/// A point is in P exactly if its shift is in P - v.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    const Polytope P = test::random_polytope(rng, 3, 5);
    const Tensor v = column(test::random_direction(rng, 3));
    const Polytope shifted = P.minus(v);
    for (int k = 0; k < 30; ++k) {
        std::vector<double> x = test::random_direction(rng, 3);
        const Tensor p = column(x);
        same = same && P.contains(p) == shifted.contains(p - v);
    }
    check(same, b + ": x in P exactly if x - v in P - v");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        shifts(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().minus(Tensor::zeros({3, 1})); },
                       Error::dimension_mismatch),
          "minus: the vector has the dimension");
    ChecksScope off(false);
    check(close(test::square_h().minus(column({1.0, 1.0})).b(), {0.0, 0.0, 1.0, 1.0}),
          "minus: the same with the checks off");
    return test::finish("polytope minus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
