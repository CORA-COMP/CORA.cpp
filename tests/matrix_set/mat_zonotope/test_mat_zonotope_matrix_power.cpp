// test_mat_zonotope_matrix_power - MatZonotope matrix_power: the power of a square set
//
// CORA: matrixSet/@matZonotope/mpower.m
// PROPERTY: matrixSet/@matZonotope/mpower.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The exponent 0 gives the identity without generators, 1 the set itself, 2 the product A A and
/// 3 the product (A A) A.
void powers_of_the_example(const std::string &b) {
    const MatZonotope A = test::example();
    const MatZonotope P0 = A.matrix_power(0);
    check(close(P0.center(), {1, 0, 0, 1}) && P0.num_generators() == 0, b + ": the identity for 0");
    check(test::same_set(A.matrix_power(1), A, 0), b + ": the set for 1");
    check(test::same_set(A.matrix_power(2), A.matmul(A), 0), b + ": A A for 2");
    check(test::same_set(A.matrix_power(3), A.matmul(A).matmul(A), 0), b + ": (A A) A for 3");
    check(A.matrix_power(2).num_generators() == 8 && A.matrix_power(3).num_generators() == 26,
          b + ": 3^k - 1 generators");
}

/// The power of every member is in the box of the power, and a point matrix gives its power; a
/// batch is per set.
void powers_enclose_the_members(const std::string &b) {
    Rng rng(31);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.4);
    for (const int64_t k : {2, 3, 4}) {
        const IntervalMatrix box = A.matrix_power(k).interval_matrix();
        bool inside = true;
        for (int trial = 0; trial < 20; ++trial) {
            Tensor M = test::member(A, test::coefficients(rng, 2)), power = M;
            for (int64_t i = 1; i < k; ++i) power = power.matmul(M);
            inside = inside && box.contains(power, "exact", 1e-12);
        }
        check(inside, b + ": the k-th power of every member is inside, k = " + std::to_string(k));
    }
    const MatZonotope point = MatZonotope(A.center()).matrix_power(3);
    check(close(point.center(), A.center().matmul(A.center()).matmul(A.center()), 1e-12),
          b + ": a point matrix");
    const MatZonotope batch = test::batch_of(A, A).matrix_power(0);
    check(batch.center().shape() == std::vector<int64_t>({2, 3, 3}), b + ": a batch of identities");
    const MatZonotope squares = test::batch_of(A, A).matrix_power(2);
    // a batch of squares
    check(test::same_set(test::batch_member(squares, 1), A.matrix_power(2), 1e-12),
          b + ": a batch of squares");
}

/// A non-square set and a negative exponent are refused; with the checks off the same power.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { MatZonotope(test::matrix({1, 2, 3}, 1, 3)).matrix_power(2); },
                       Error::dimension_mismatch),
          "matrix_power: a non-square set");
    check(test::raises([&] { A.matrix_power(-1); }, Error::wrong_value),
          "matrix_power: a negative exponent");
    const Tensor on = A.matrix_power(2).center();
    ChecksScope off(false);
    check(close(A.matrix_power(2).center(), on, 0), "matrix_power: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        powers_of_the_example(b);
        powers_enclose_the_members(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope matrix_power");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
