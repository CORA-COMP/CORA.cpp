// test_interval_matmul - Interval matmul: the product with sets, matrices and numbers
//
// CORA: matrixSet/@intervalMatrix/mtimes.m
// PROPERTY: matrixSet/@intervalMatrix/mtimes.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/zonotope/zonotope.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// By hand with A = [[[1, 2], [-1, 1]], [[0, 3], [2, 2]]] and M = [[1, 0], [2, -1]].
void products_with_a_matrix(const std::string &b) {
    const Interval A = test::example();
    const Tensor M = test::matrix({1, 0, 2, -1}, 2, 2);
    const Interval right = A.matmul(M), left = (M * A);
    check(close(right.inf(), std::vector<double>{-1, -1, 4, -2}) &&
              close(right.sup(), std::vector<double>{4, 1, 7, -2}),
          b + ": A * M");
    check(close(left.inf(), std::vector<double>{1, -1, -1, -4}) &&
              close(left.sup(), std::vector<double>{2, 1, 4, 0}),
          b + ": M * A");
    const Interval scaled = (-2.0 * A);
    check(close(scaled.inf(), std::vector<double>{-4, -2, -6, -4}) &&
              close(scaled.sup(), std::vector<double>{-2, 2, 0, -4}),
          b + ": a negative number swaps the bounds");
    check(close((A * 2.0).sup(), std::vector<double>{4, 2, 6, 4}), b + ": A * 2");
}

/// The product of two sets is the sum of the hulls of the corner products; a point factor is exact.
void product_of_two_sets(const std::string &b) {
    const Interval A = Interval::matrix(test::matrix({1, -1}, 1, 2), test::matrix({2, 1}, 1, 2));
    const Interval B = Interval::matrix(test::matrix({-1, 3}, 2, 1), test::matrix({2, 4}, 2, 1));
    // [1, 2] [-1, 2] + [-1, 1] [3, 4] = [-2, 4] + [-4, 4] = [-6, 8]
    const Interval P = (A * B);
    check(close(P.inf(), std::vector<double>{-6}) && close(P.sup(), std::vector<double>{8}),
          b + ": a row times a column");
    Rng rng(12);
    const Interval C = test::random_matrix(rng, 3, 0.4), D = test::random_matrix(rng, 3, 0.4);
    const Interval CD = C * D;
    bool inside = true;
    for (int trial = 0; trial < 30; ++trial)
        inside = inside && CD.contains(test::member(C, rng).matmul(test::member(D, rng)), 1e-12);
    check(inside, b + ": products of members are inside");
    const Tensor M = test::member(C, rng);
    check(close((test::interval_matrix(M) * test::interval_matrix(M)).center(), M.matmul(M), 1e-12),
          b + ": point times point");
}

/// A set times a zonotope encloses the products; the empty set maps to the empty set.
void product_with_other_sets(const std::string &b) {
    const Interval A = test::example();
    const Zonotope Z(test::vec({1, 1}), test::matrix({1, 0, 0, 1}, 2, 2));
    const Zonotope R = A * Z;
    check(close(R.center().expand_dims(-1), std::vector<double>{1.5, 3.5}), b + ": the center T c");
    check(
        close(R.generators().matrix_transpose(), test::matrix({1.5, 0, 3, 0, 1.5, 2, 0, 3}, 2, 4)),
        b + ": the generators [T G, diag(R (|c| + sum|G|))]");
    // a member times a point of Z, the point as CORA's column (2, 1)
    Rng rng(13);
    bool inside = true;
    for (int trial = 0; trial < 30; ++trial) {
        const Tensor x = Z.rand_point(1, rng).matrix_transpose();
        inside = inside && R.contains(test::member(A, rng).matmul(x).squeeze(-1), "exact", 1e-12);
    }
    check(inside, b + ": the products of members and points are inside");
    check(EmptySet(2).linear_map(A).dim() == 2, b + ": the empty set stays empty");
    const Tensor six = test::matrix({1, 2, 3, 4, 5, 6}, 3, 2);
    const Interval tall = test::interval_matrix(six);
    check(EmptySet(2).linear_map(tall).dim() == 3, b + ": its dimension is the number of rows");
}

/// Inner dimensions that differ are refused; with the checks off a valid product is the same.
void checks_and_their_switch() {
    const Interval A = test::example();
    const Tensor wrong = test::matrix({1, 2, 3}, 3, 1);
    check(test::raises([&] { A.matmul(wrong); }, Error::dimension_mismatch),
          "matmul: a matrix of 3 rows");
    check(test::raises([&] { A.linear_map(wrong); }, Error::dimension_mismatch),
          "matmul: M * A with M of 1 column");
    check(test::raises([&] { A.matmul(test::interval_matrix(wrong)); }, Error::dimension_mismatch),
          "matmul: a set of 3 rows");
    const Tensor on = A.matmul(A).sup();
    ChecksScope off(false);
    check(close(A.matmul(A).sup(), on, 0), "matmul: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        products_with_a_matrix(b);
        product_of_two_sets(b);
    });
    test::for_each_backend(product_with_other_sets);
    checks_and_their_switch();
    return test::finish("interval_matrix matmul");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
