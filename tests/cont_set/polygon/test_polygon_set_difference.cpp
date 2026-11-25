// test_polygon_set_difference - Polygon::set_difference: every piece cut by the other's pieces
//
// CORA: contSet/@polygon/subtract.m
// PROPERTY: contSet/@polygon/subtract.m
// CORA: contSet/@polygon/mldivide.m
// PROPERTY: contSet/@polygon/mldivide.m
// CORA authors: Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polygon_testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The share of N random points of [-4, 4]^2 where the polygon R answers as `in` says.
template <class F>
bool aux_classifies(const Polygon &R, Rng &rng, F in) {
    const Tensor X = test::random_points(rng, 300, -4, 4);
    for (int i = 0; i < 300; ++i)
        if (R.contains(test::row(X, i)) != in(test::row(X, i))) return false;
    return true;
}

/// Stated properties: the parts are disjoint and lie in A, the areas add up (|A \ B| = |A| -
/// |A n B|), and random points are in it iff in A and not in B (off the boundaries).
void defining_properties() {
    Rng rng(23);
    for (int trial = 0; trial < 15; ++trial) {
        const Polygon A = test::random_polygon(rng, 2), B = test::random_polygon(rng, 2);
        const Polygon R = A.set_difference(B);
        double sum = 0;
        for (const Polytope &P : R.pieces()) sum += P.volume();
        check(close(sum, R.volume(), 1e-9), "set_difference: the parts do not overlap");
        check(close(R.volume(), A.volume() - A.intersect(B).volume(), 1e-9),
              "set_difference: |A \\ B| = |A| - |A n B|");
        check(aux_classifies(R, rng,
                             [&](const Tensor &x) { return A.contains(x) && !B.contains(x); }),
              "set_difference: random points are in it iff in A and not in B");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    defining_properties();
    check(close(test::l_shape().set_difference(test::unit_square()).volume(), 2.0),
          "set_difference: the L without the square");
    // an invalid operand is refused before the operation
    check(test::raises([] { Polygon(Tensor::from_data({0.0, 0.0, 0.0}, {1, 3})); },
                       Error::wrong_input_in_constructor),
          "set_difference: an operand of three columns is refused");
    {
        ChecksScope off(false);
        check(close(test::l_shape().set_difference(test::unit_square()).volume(), 2.0),
              "set_difference: the same with the checks off");
    }
    return test::finish("polygon set_difference");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
