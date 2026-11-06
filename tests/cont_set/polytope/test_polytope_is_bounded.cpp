// test_polytope_is_bounded - Polytope::is_bounded: whether the set is bounded
//
// CORA: contSet/@polytope/isBounded.m
// PROPERTY: contSet/@polytope/isBounded.m
//
// Property: the random polytopes with the box |x| <= 3 are bounded, a halfspace is not
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Bounded and unbounded sets of both forms and in one dimension.
void boundedness(const std::string &b) {
    check(test::square_h().is_bounded() && test::square_v().is_bounded(), b + ": the square");
    check(test::triangle_h().is_bounded(), b + ": the triangle");
    // one halfspace, a strip and R^n are unbounded
    const Polytope halfspace(Tensor({{1.0, 0.0}}), column({1.0}));
    const Polytope strip(Tensor({{1.0, 0.0}, {-1.0, 0.0}}), column({1.0, 1.0}));
    check(!halfspace.is_bounded() && !strip.is_bounded() && !Polytope::inf(2).is_bounded(),
          b + ": a halfspace, a strip and R^n");
    check(Polytope::empty(2).is_bounded(), b + ": the empty set is bounded");
    // one dimension: the vertices and the rows
    const double inf = std::numeric_limits<double>::infinity();
    check(!Polytope(Tensor::from_data({0.0, inf}, {1, 2})).is_bounded(),
          b + ": 1D: a vertex at Inf");
    check(Polytope(Tensor::from_data({0.0, 1.0}, {1, 2})).is_bounded(), b + ": 1D: a segment");
    check(!Polytope(Tensor({{-1.0}}), column({0.0})).is_bounded(), b + ": 1D: a half-line");
    check(Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 1.0})).is_bounded(), b + ": 1D: [0, 1]");
    // a batch of vertex sets
    check(Polytope(Tensor::ones({3, 2, 4})).is_bounded(), b + ": a batch of vertices");
}

/// Random polytopes are bounded, and so their support function is finite along any direction.
void properties(const std::string &b) {
    Rng rng(4);
    bool bounded = true, finite = true;
    for (int n = 2; n <= 4; ++n) {
        const Polytope P = test::random_polytope(rng, n, 2 * n);
        bounded = bounded && P.is_bounded();
        finite = finite && std::isfinite(test::support(P, test::random_direction(rng, n)));
    }
    check(bounded && finite, b + ": random polytopes with the box |x| <= 3 are bounded");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        boundedness(b);
        properties(b);
    });
    // a batch of halfspaces needs a program per member
    check(test::raises(
              [] { Polytope(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2, 1})).is_bounded(); },
              Error::not_supported),
          "is_bounded: halfspaces of a batch");
    ChecksScope off(false);
    check(test::square_h().is_bounded(), "is_bounded: the same with the checks off");
    return test::finish("polytope is_bounded");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
