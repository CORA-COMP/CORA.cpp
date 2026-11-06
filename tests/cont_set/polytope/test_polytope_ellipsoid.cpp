// test_polytope_ellipsoid - Polytope::ellipsoid: an ellipsoid around a bounded polytope
//
// CORA: contSet/@polytope/ellipsoid.m
// PROPERTY: contSet/@polytope/ellipsoid.m
//
// Property: the ellipsoid contains the vertices
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

/// The covariance ellipsoid of a square and of a point, and the empty set.
void ellipsoids(const std::string &b) {
    const Ellipsoid E = test::square_h().ellipsoid();
    check(close(E.q(), {0.5, 0.5}), b + ": centered at the mean");
    check(E.contains(test::square_v().V(), "exact", 1e-8), b + ": it contains the corners");
    check(test::square_v().ellipsoid().contains(test::square_v().V(), "exact", 1e-8),
          b + ": vertices");
    check(Polytope::empty(2).ellipsoid().dim() == 2, b + ": the empty set");
}

/// The ellipsoid contains the vertices of random polytopes.
void properties(const std::string &b) {
    Rng rng(4);
    bool contained = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 2);
        contained = contained && P.ellipsoid().contains(test::vertices_of(P), "exact", 1e-7);
    }
    check(contained, b + ": it contains the vertices");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        ellipsoids(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().ellipsoid("big"); }, Error::wrong_value),
          "ellipsoid: the method");
    check(test::raises([] { test::square_h().ellipsoid("inner"); }, Error::not_supported),
          "ellipsoid: the semidefinite programs are not ported");
    check(test::raises([] { Polytope(Tensor({{1.0, 0.0}}), column({1.0})).ellipsoid(); },
                       Error::special_error),
          "ellipsoid: unbounded");
    ChecksScope off(false);
    check(test::square_h().ellipsoid().dim() == 2, "ellipsoid: the same with the checks off");
    return test::finish("polytope ellipsoid");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
