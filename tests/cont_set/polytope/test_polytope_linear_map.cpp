// test_polytope_linear_map - Polytope::linear_map: the image under a matrix
//
// CORA: contSet/@polytope/mtimes.m
// PROPERTY: contSet/@polytope/mtimes.m
//
// Property: M x is in the image of P exactly if x is in P (invertible M); samples map into the
// image
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

/// Vertices map, a scalar scales, an invertible matrix divides the rows, the zero matrix gives the
/// origin.
void images(const std::string &b) {
    const Tensor M({{2.0, 0.0}, {0.0, 3.0}});
    check(close((M * test::square_v()).V(), Tensor({{0.0, 2.0, 2.0, 0.0}, {0.0, 0.0, 3.0, 3.0}})),
          b + ": vertices");
    const Polytope image = M * test::square_h();
    check(image.contains(column({2.0, 3.0})) && !image.contains(column({2.1, 3.0})) &&
              close(test::support(image, {1.0, 1.0}), 5.0, 1e-8),
          b + ": halfspaces, invertible M");
    // a scalar: positive scales the offsets, negative flips the rows too
    const Polytope scaled =
        Polytope(Tensor({{-1.0}, {1.0}}), column({0.0, 1.0})).linear_map(Tensor({{-2.0}}));
    check(scaled.contains(column({-2.0})) && scaled.contains(column({0.0})) &&
              !scaled.contains(column({0.5})),
          b + ": 1D: scaling by a negative number");
    check(close(test::square_h().linear_map(Tensor::zeros({2, 2})).V(), {0.0, 0.0}),
          b + ": the zero matrix: the origin");
}

/// Non-square and singular matrices: the projection and the lifting of the decomposition.
void decomposed(const std::string &b) {
    // a wide matrix drops the second coordinate: the projection onto x
    const Polytope shadow = Tensor({{1.0, 0.0}}) * test::triangle_h();
    check(shadow.dim() == 1 && shadow.contains(column({2.0})) && !shadow.contains(column({2.1})) &&
              shadow.contains(column({0.0})) && !shadow.contains(column({-0.1})),
          b + ": wide: the shadow of the triangle");
    // a tall matrix embeds the plane in space: the equality z = x + y holds
    const Polytope lifted = Tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}) * test::square_h();
    check(lifted.dim() == 3 && lifted.contains(column({1.0, 1.0, 2.0})) &&
              !lifted.contains(column({1.0, 1.0, 3.0})) &&
              !lifted.contains(column({1.0, 1.1, 2.1})),
          b + ": tall: the plane z = x + y");
    // a singular matrix, x -> (x + y, x + y), maps the square onto the diagonal segment [0, 2]
    const Polytope segment = Tensor({{1.0, 1.0}, {1.0, 1.0}}) * test::square_h();
    check(segment.contains(column({1.0, 1.0})) && segment.contains(column({2.0, 2.0})) &&
              !segment.contains(column({1.0, 1.5})) && !segment.contains(column({2.1, 2.1})),
          b + ": singular: a diagonal segment");
    // R^n: the whole space for a matrix of full row rank, else its range
    check((Tensor({{1.0, 0.0}}) * Polytope::inf(2)).representsa("fullspace"), b + ": R^n: R^m");
}

/// A batch of matrices on vertices.
void batches(const std::string &b) {
    const Tensor M = Tensor::stack({Tensor::eye(2), Tensor::eye(2) * 2.0}, 0);
    const Polytope image = test::square_v().linear_map(M);
    check(image.V().shape() == Shape{2, 2, 4} && close(image.V().slice(0, 1, 2).max(), {2.0}),
          b + ": a batch of matrices");
}

/// Samples of the set map into the image; the inverse map brings the image back.
void properties(const std::string &b) {
    Rng rng(4);
    bool inside = true, back = true;
    for (int n = 2; n <= 3; ++n) {
        const Polytope P = test::random_polytope(rng, n, n + 2);
        std::vector<double> m(std::size_t(n * n));
        rng.normal(m.data(), m.size(), 1.0);
        const Tensor M = Tensor::from_data(m, {n, n}) + Tensor::eye(n) * 3.0;
        const Polytope image = P.linear_map(M);
        const Tensor x = test::sample_points(P, rng, 10);
        inside = inside && image.contains(M.matmul(x), "exact", 1e-7);
        back = back && image.linear_map(M.inv()).isequal(P, 1e-6);
    }
    check(inside && back, b + ": samples map into the image, and M^-1 M P = P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        images(b);
        decomposed(b);
        batches(b);
        properties(b);
    });
    // the columns of the matrix must be the dimension; a batch of singular matrices has no rows
    check(test::raises([] { test::square_h().linear_map(Tensor::eye(3)); },
                       Error::dimension_mismatch),
          "linear_map: the matrix has the columns of the dimension");
    check(test::raises(
              [] {
                  test::square_h().linear_map(
                      Tensor::stack({Tensor({{1.0, 1.0}}), Tensor({{1.0, 1.0}})}, 0));
              },
              Error::not_supported),
          "linear_map: a batch of non-invertible matrices on halfspaces");
    ChecksScope off(false);
    check(test::square_v().linear_map(Tensor::eye(2) * 2.0).V().max().item() == 2.0,
          "linear_map: the same with the checks off");
    return test::finish("polytope linear_map");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
