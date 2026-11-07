// test_polytope_is_full_dim - Polytope::is_full_dim: whether the set has an interior
//
// CORA: contSet/@polytope/isFullDim.m
// PROPERTY: contSet/@polytope/isFullDim.m
//
// Property: the random polytopes with the box |x| <= 3 have an interior, in both forms
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

/// Full-dimensional and flat sets of both forms.
void interiors(const std::string &b) {
    check(test::square_h().is_full_dim() && test::square_v().is_full_dim(), b + ": the square");
    check(test::triangle_h().is_full_dim() && Polytope::inf(2).is_full_dim(),
          b + ": triangle, R^n");
    // a segment, a point and the empty set have none
    const Polytope segment(Tensor({{0.0, 1.0}, {0.0, 1.0}}));
    check(!segment.is_full_dim() && !Polytope(column({1.0, 2.0})).is_full_dim(),
          b + ": segment, point");
    check(!Polytope::empty(2).is_full_dim(), b + ": the empty set");
    // an equality leaves no interior, nor do two opposite rows that touch
    const Polytope equality(test::square_h().A(), test::square_h().b(), Tensor({{1.0, -1.0}}),
                            column({0.0}));
    const Polytope flat(Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
                        column({1.0, -1.0, 1.0, 1.0}));
    check(!equality.is_full_dim() && !flat.is_full_dim(), b + ": an equality, two touching rows");
    // one dimension: two vertices, a point, a half-line
    check(Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 1.0})).is_full_dim(), b + ": 1D: [0, 1]");
    check(!Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 0.0})).is_full_dim(),
          b + ": 1D: a point");
    check(Polytope(Tensor({{-1.0}}), column({0.0})).is_full_dim(), b + ": 1D: a half-line");
}

/// The subspace: the identity for a full-dimensional set, else a basis of the span.
void subspaces(const std::string &b) {
    const auto [full, X] = test::square_h().is_full_dim_subspace();
    check(full && close(X, Tensor::eye(2), 0), b + ": full: the identity");
    // the diagonal segment spans (1, 1)
    const Polytope diagonal(Tensor({{0.0, 1.0}, {0.0, 1.0}}));
    const auto [vfull, V] = diagonal.is_full_dim_subspace();
    check(!vfull && V.shape() == Shape{2, 1}, b + ": vertices: one direction");
    check(std::abs(std::abs(V.data()[0]) - std::abs(V.data()[1])) < 1e-9,
          b + ": the direction (1, 1)");
    const Polytope on_diagonal(test::square_h().A(), test::square_h().b(), Tensor({{1.0, -1.0}}),
                               column({0.0}));
    const auto [hfull, H] = on_diagonal.is_full_dim_subspace();
    check(!hfull && H.shape() == Shape{2, 1} &&
              std::abs(std::abs(H.data()[0]) - std::abs(H.data()[1])) < 1e-6,
          b + ": halfspaces: the direction (1, 1)");
    check(Polytope::empty(2).is_full_dim_subspace().second.shape() == Shape{2, 0},
          b + ": the empty set has no direction");
}

/// Random polytopes have an interior, in either form.
void properties(const std::string &b) {
    Rng rng(4);
    bool full = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, 2 * n);
        full = full && P.is_full_dim() && test::as_vertices(P).is_full_dim();
    }
    check(full, b + ": random polytopes have an interior");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        interiors(b);
        subspaces(b);
        properties(b);
    });
    // halfspaces of a batch need a program per member; a negative tolerance is refused
    check(test::raises(
              [] { Polytope(Tensor::ones({3, 2, 2}), Tensor::ones({3, 2, 1})).is_full_dim(); },
              Error::not_supported),
          "is_full_dim: halfspaces of a batch");
    check(test::raises([] { test::square_h().is_full_dim(-1.0); }, Error::wrong_value),
          "is_full_dim: tol >= 0");
    ChecksScope off(false);
    check(test::square_h().is_full_dim(), "is_full_dim: the same with the checks off");
    return test::finish("polytope is_full_dim");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
