// test_polytope_project_high_dim - Polytope::project_high_dim: a set in a larger space
//
// CORA: contSet/@polytope/projectHighDim.m
// PROPERTY: contSet/@polytope/projectHighDim.m
//
// Property: a point is in the embedding exactly if the other coordinates are zero and the rest in
// the set
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

/// Halfspaces get equalities for the new coordinates, vertices are padded with zeros.
void embeddings(const std::string &b) {
    const Polytope H = test::square_h().project_high_dim(3, {2, 0});
    check(H.dim() == 3 && close(H.Ae(), {0, 1, 0}, 0) && close(H.be(), {0}, 0),
          b + ": halfspaces: the equality for the new coordinate");
    check(H.contains(column({0.5, 0.0, 0.5})) && !H.contains(column({0.5, 0.1, 0.5})),
          b + ": halfspaces: the points");
    const Polytope V = test::square_v().project_high_dim(3, {2, 0});
    check(close(V.V(), {0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0}), b + ": vertices: the padding");
    // N = n only moves the coordinates
    check(close(test::square_v().project_high_dim(2, {1, 0}).V(), {0, 0, 1, 1, 0, 1, 1, 0}),
          b + ": N = n: a permutation");
    check(test::square_h().project_high_dim(2, {1, 0}).Ae().shape() == Shape{0, 2},
          b + ": N = n: no new equality");
}

/// The embedding holds exactly the points with zeros elsewhere.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    const Polytope P = test::random_polytope(rng, 2, 4);
    const Polytope E = P.project_high_dim(4, {3, 1});
    for (int k = 0; k < 40; ++k) {
        std::vector<double> x = test::random_direction(rng, 2);
        const Tensor in = column({0.0, x[1], 0.0, x[0]}), off = column({0.1, x[1], 0.0, x[0]});
        same = same && E.contains(in, "exact", 1e-9) == P.contains(column({x[0], x[1]})) &&
               !E.contains(off, "exact", 1e-9);
    }
    check(same, b + ": the embedding holds the points with zeros elsewhere");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        embeddings(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().project_high_dim(1, {0, 1}); }, Error::wrong_value),
          "project_high_dim: the larger space is not smaller");
    check(test::raises([] { test::square_h().project_high_dim(3, {1, 1}); }, Error::wrong_value),
          "project_high_dim: a repeated position");
    ChecksScope off(false);
    check(test::square_h().project_high_dim(3, {0, 1}).dim() == 3,
          "project_high_dim: the same with the checks off");
    return test::finish("polytope project_high_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
