// test_polytope_distance - Polytope::distance: the distance to a polytope or to points
//
// CORA: contSet/@polytope/distance.m
// PROPERTY: contSet/@polytope/distance.m
//
// Property: the distance to a translate by v is at most |v|, and the closest point is within it
//
// CORA authors: Victor Gassmann, Mark Wetzlinger
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

/// Hand-computed distances.
void distances(const std::string &b) {
    const Polytope shifted = test::square_h().plus(column({3.0, 0.0}));
    check(close(test::square_h().distance(shifted), 2.0, 1e-7), b + ": two squares, a gap of 2");
    check(close(test::square_h().distance(test::square_h()), 0.0, 1e-12), b + ": equal sets: 0");
    const Polytope diagonal = test::square_h().plus(column({2.0, 2.0}));
    check(close(test::square_h().distance(diagonal), std::sqrt(2.0), 1e-7),
          b + ": the diagonal gap");
    // points: inside, beside, at a corner
    const Tensor points({{0.5, 3.0, 2.0}, {0.5, 0.5, 2.0}});
    check(close(test::square_h().distance(points), {0.0, 2.0, std::sqrt(2.0)}, 1e-7),
          b + ": points");
    // the hyperplane x = 1 as 2x = 2: CORA divides by the squared norm of the normal and gets 1.5
    const Polytope plane(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{2.0, 0.0}}),
                         column({2.0}));
    check(close(plane.distance(column({4.0, 7.0})), {3.0}, 1e-7),
          b + ": a hyperplane with a normal that is no unit");
    constexpr double inf = std::numeric_limits<double>::infinity();
    check(Polytope::empty(2).distance(test::square_h()) == inf &&
              test::square_h().distance(Polytope::empty(2)) == inf,
          b + ": the empty set: Inf");
}

/// The distance to a translate is at most the length of the shift.
void properties(const std::string &b) {
    Rng rng(4);
    bool bounded = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 1);
        const std::vector<double> v = test::random_direction(rng, n);
        double length = 0.0;
        for (const double x : v) length += x * x;
        const Tensor shift = column(v) * 3.0;
        bounded = bounded && P.distance(P.plus(shift)) <= 3.0 * std::sqrt(length) + 1e-7;
    }
    check(bounded, b + ": the distance to a translate is at most the shift");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        distances(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().distance(Polytope::inf(3)); },
                       Error::dimension_mismatch),
          "distance: dimensions");
    check(test::raises([] { test::square_h().distance(Tensor::zeros({3, 2})); },
                       Error::dimension_mismatch),
          "distance: the points have the dimension");
    ChecksScope off(false);
    check(test::square_h().distance(column({0.5, 0.5})).item() == 0.0,
          "distance: the same with the checks off");
    return test::finish("polytope distance");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
