// test_polygon_intersect - Polygon::intersect: every pair of pieces clipped
//
// CORA: contSet/@polygon/and_.m
// PROPERTY: contSet/@polygon/and_.m
// CORA authors: Niklas Kochdumper
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

/// Stated properties: every piece lies in both operands, the areas add up (|A n B| + |A \ B| =
/// |A|), and random points are in it iff they are in both.
void defining_properties() {
    Rng rng(19);
    for (int trial = 0; trial < 15; ++trial) {
        const Polygon A = test::random_polygon(rng, 2), B = test::random_polygon(rng, 2);
        const Polygon R = A.intersect(B);
        bool inside = true;
        for (const Polytope &P : R.pieces()) {
            // the vertices pulled toward the piece's mean by a relative 1e-9, off the rounding
            const Tensor c = P.V().mean(0, true);
            const Tensor V = c + (P.V() - c) * (1 - 1e-9);
            inside = inside && A.contains(V) && B.contains(V);
        }
        check(inside, "intersect: every piece lies in both operands");
        check(close(R.volume() + A.set_difference(B).volume(), A.volume(), 1e-9),
              "intersect: |A n B| + |A \\ B| = |A|");
        check(
            aux_classifies(R, rng, [&](const Tensor &x) { return A.contains(x) && B.contains(x); }),
            "intersect: random points are in it iff in both");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    defining_properties();
    check(close(test::l_shape().intersect(test::unit_square()).volume(), 1.0) &&
              (Set(test::l_shape()) & Set(test::unit_square())).as<Polygon>().num_pieces() >= 1,
          "intersect: the square inside the L, the operator &");
    // an invalid operand is refused before the operation
    check(test::raises([] { Polygon(Tensor::from_data({0.0, 0.0, 0.0}, {1, 3})); },
                       Error::wrong_input_in_constructor),
          "intersect: an operand of three columns is refused");
    {
        ChecksScope off(false);
        check(close(test::l_shape().intersect(test::unit_square()).volume(), 1.0),
              "intersect: the same with the checks off");
    }
    return test::finish("polygon intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
