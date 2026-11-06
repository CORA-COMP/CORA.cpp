// test_polytope_box - Polytope::box: the polytope of the tightest box
//
// CORA: contSet/@polytope/box.m
// PROPERTY: contSet/@polytope/box.m
//
// Property: the result is the polytope of the box of the set, and contains the set
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
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

/// The box of a triangle is its bounding square, as halfspaces.
void box_of(const std::string &b) {
    const Polytope tri(Tensor({{-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}), column({0.0, 0.0, 2.0}));
    const Polytope B = tri.box();
    check(B.is_h_rep() && close(B.A(), {1, 0, 0, 1, -1, 0, 0, -1}, 0), b + ": rows [I; -I]");
    check(close(B.b(), {2, 2, 0, 0}, 1e-9), b + ": offsets [sup; -inf]");
    check(B.contains(column({2.0, 2.0})) && !B.contains(column({2.1, 0.0})), b + ": the corners");
    const Polytope V = Polytope(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}})).box();
    check(close(V.b(), {1, 1, 0, 0}, 0), b + ": vertices: the same box");
    check(Polytope::empty(2).box().interval().is_empty(), b + ": the empty set");
}

/// A checks-off call.
void refusals() {
    const Polytope P(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    ChecksScope off(false);
    check(P.box().A().shape() == Shape{4, 2}, "box: the same with the checks off");
    check(test::raises([&] { Polytope(Tensor::ones({2, 1, 2}), Tensor::ones({2, 1, 1})).box(); },
                       Error::not_supported),
          "box: a batch of halfspace polytopes");
}

/// The box of a random polytope holds its samples, and its box is the polytope's box.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3}) {
        const Polytope Q = test::random_polytope(rng, n, 2 * n);
        const Polytope B = Q.box();
        check(test::inside_halfspaces(test::sample_points(Q, rng, 30), B.constraints()),
              b + ": the box holds the samples, " + std::to_string(n) + "D");
        check(close(B.interval().inf(), Q.interval().inf(), 1e-8) &&
                  close(B.interval().sup(), Q.interval().sup(), 1e-8),
              b + ": the box of the box, " + std::to_string(n) + "D");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        box_of(b);
        properties(b);
    });
    refusals();
    return test::finish("polytope box");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
