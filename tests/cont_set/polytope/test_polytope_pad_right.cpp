// test_polytope_pad_right - Polytope::pad_right: new free coordinates before the old
//
// The rows of the unit square and of equalities gain zero columns on the left, the offsets stay;
// a batch pads each member. The stated property: (y, x) is in P.pad_right(k) exactly if x is in
// P, whatever y.
//
// CORA.cpp only: tests CORALean's HPolytope.padRight
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

/// The rows of the square and of equalities with zeros on the left.
void rows(const std::string &b) {
    const Polytope P = test::square_h().pad_right(1);
    check(close(P.A(), {0, 1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1}, 0.0) &&
              close(P.b(), {1, 1, 0, 0}, 0.0) && P.dim() == 3,
          b + ": the square");
    const Polytope E = Polytope::equalities(Tensor({{1.0, 2.0}}), vec({3.0})).pad_right(2);
    check(close(E.Ae(), {0, 0, 1, 2}, 0.0) && close(E.be(), {3}, 0.0) &&
              E.A().shape() == Shape{0, 4},
          b + ": equalities");
    // a batch of two members pads each
    const Polytope B(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1}));
    check(B.pad_right(1).A().shape() == Shape{2, 1, 3}, b + ": a batch");
}

/// (y, x) in P.pad_right(k) exactly if x in P.
void properties(const std::string &b) {
    Rng rng(8);
    const Polytope P = test::random_polytope(rng, 2, 4), padded = P.pad_right(2);
    bool same = true;
    for (int k = 0; k < 30; ++k) {
        const std::vector<double> x = test::random_direction(rng, 2);
        const std::vector<double> y = test::random_direction(rng, 2);
        same = same && padded.contains(vec({9 * y[0], 9 * y[1], x[0], x[1]})) == P.contains(vec(x));
    }
    check(same, b + ": the new coordinates are free");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rows(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().pad_right(-1); }, Error::wrong_value),
          "pad_right: a negative count");
    ChecksScope off(false);
    check(test::square_h().pad_right(2).dim() == 4, "pad_right: the same with the checks off");
    return test::finish("polytope pad_right");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
