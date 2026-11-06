// test_polytope_support_function - Polytope::support_function: the largest value along d
//
// CORA: contSet/@polytope/supportFunc_.m
// CORA: contSet/@polytope/private/priv_supportFunc.m
// PROPERTY: contSet/@polytope/supportFunc_.m
//
// Property: the support function dominates every sample of the set and is attained at a vertex
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

/// Hand-computed values in both forms, and the special sets.
void values(const std::string &b) {
    const Tensor d = column({1.0, 2.0});
    check(close(test::square_h().support_function(d), {3.0}), b + ": halfspaces: the square");
    check(close(test::square_v().support_function(d), {3.0}), b + ": vertices: the square");
    check(close(test::triangle_h().support_function(column({1.0, 1.0})), {2.0}),
          b + ": the triangle along (1, 1)");
    constexpr double inf = std::numeric_limits<double>::infinity();
    check(Polytope::inf(2).support_function(d).item() == inf, b + ": R^n: Inf");
    check(Polytope::inf(2).support_function(column({0.0, 0.0})).item() == 0.0,
          b + ": R^n along 0: 0");
    check(Polytope::empty(2).support_function(d).item() == -inf, b + ": the empty set: -Inf");
    check(Polytope(Tensor::zeros({2, 0})).support_function(d).item() == -inf,
          b + ": no vertices: -Inf");
    // unbounded along the direction of a halfspace, finite against it
    const Polytope halfspace(Tensor({{1.0, 0.0}}), column({1.0}));
    check(halfspace.support_function(column({0.0, 1.0})).item() == inf &&
              close(halfspace.support_function(column({1.0, 0.0})), {1.0}),
          b + ": a halfspace");
    // a batch of vertex sets
    const Tensor V({{0.0, 1.0}, {0.0, 1.0}});
    const Polytope batch(Tensor::stack({V, V * 2.0}, 0));
    check(close(batch.support_function(Tensor::stack({d, d}, 0)), {3.0, 6.0}), b + ": a batch");
}

/// Sampled points stay below the support function, which is reached at a vertex.
void properties(const std::string &b) {
    Rng rng(4);
    bool below = true, attained = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, 2 * n);
        const Tensor points = test::sample_points(P, rng, 30);
        const std::vector<double> d = test::random_direction(rng, n);
        const double h = test::support(P, d);
        const std::vector<double> values = column(d).matrix_transpose().matmul(points).data();
        for (const double v : values) below = below && v <= h + 1e-9;
        // the vertex form agrees
        attained = attained && close(test::support(test::as_vertices(P), d), h, 1e-8);
    }
    check(below && attained,
          b + ": the support function bounds the points and the vertices attain it");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        values(b);
        properties(b);
    });
    // a linear program per direction: halfspaces take a single direction
    check(test::raises([] { test::square_h().support_function(Tensor::ones({2, 2, 1})); },
                       Error::not_supported),
          "support_function: halfspaces and several directions");
    ChecksScope off(false);
    check(close(test::square_v().support_function(column({1.0, 1.0})), {2.0}),
          "support_function: the same with the checks off");
    return test::finish("polytope support_function");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
