// test_polytope_half_scaled_left - Polytope::half_scaled_left: the rows of a set at a hull weight
//
// The rows [A, 0_k, -b/2] <= b/2 of the square and the same for equalities. The stated property:
// (x, y, 1) is in the result exactly if x is in P, whatever y; (x, y, -1) exactly if A x <= 0.
//
// CORA.cpp only: tests CORALean's HPolytope.halfScaledLeft
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

/// The rows of the square and of equalities.
void rows(const std::string &b) {
    const Polytope P = test::square_h().half_scaled_left(1);
    check(close(P.A(), {1, 0, 0, -0.5, 0, 1, 0, -0.5, -1, 0, 0, 0, 0, -1, 0, 0}, 0.0) &&
              close(P.b(), {0.5, 0.5, 0, 0}, 0.0) && P.dim() == 4,
          b + ": the square");
    const Polytope E = Polytope::equalities(Tensor({{1.0, 2.0}}), vec({3.0})).half_scaled_left(0);
    check(close(E.Ae(), {1, 2, -1.5}, 0.0) && close(E.be(), {1.5}, 0.0), b + ": equalities");
}

/// (x, y, 1) in the result exactly if x in P; (x, y, -1) exactly if A x <= 0.
void properties(const std::string &b) {
    Rng rng(14);
    const Polytope P = test::random_polytope(rng, 2, 3), H = P.half_scaled_left(1);
    const Polytope cone(P.A(), P.b().zeros_like());
    bool same = true;
    for (int k = 0; k < 30; ++k) {
        const std::vector<double> x = test::random_direction(rng, 2);
        same = same && H.contains(vec({x[0], x[1], 7.0, 1.0})) == P.contains(vec(x)) &&
               H.contains(vec({x[0], x[1], -7.0, -1.0})) == cone.contains(vec(x));
    }
    check(same, b + ": the set at w = 1, its cone at w = -1");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().half_scaled_left(-1); }, Error::wrong_value),
          "half_scaled_left: a negative count");
    ChecksScope off(false);
    check(test::square_h().half_scaled_left(2).dim() == 5,
          "half_scaled_left: the same with the checks off");
    return test::finish("polytope half_scaled_left");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
