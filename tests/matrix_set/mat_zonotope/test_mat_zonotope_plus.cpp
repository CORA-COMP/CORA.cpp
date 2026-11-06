// test_mat_zonotope_plus - MatZonotope plus: the sum with a matrix zonotope or a matrix
//
// CORA: matrixSet/@matZonotope/plus.m
// PROPERTY: matrixSet/@matZonotope/plus.m
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

/// The centers add and the generators are joined, the first set's first.
void sum_of_two_sets(const std::string &b) {
    const MatZonotope A = test::example();
    const MatZonotope B = test::set_of({1, 0, 0, 1}, {0, 0, 1, 0}, 2, 2);
    const MatZonotope S = A.plus(B);
    check(close(S.center(), {2, 2, 3, 5}), b + ": the centers add");
    check(S.num_generators() == 3, b + ": the generators are joined");
    check(close(S.generators(), {1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0}),
          b + ": the first set's first");
    check(test::same_set(A + B, S), b + ": the operator");
}

/// A matrix of numbers shifts the center; a 1 x 1 number is added to every entry.
void sum_with_numbers(const std::string &b) {
    const MatZonotope A = test::example();
    const MatZonotope S = A.plus(test::matrix({1, 1, 1, 1}, 2, 2));
    check(close(S.center(), {2, 3, 4, 5}) && close(S.generators(), A.generators()),
          b + ": a matrix shifts");
    check(close(A.plus(test::matrix({10}, 1, 1)).center(), {11, 12, 13, 14}),
          b + ": a 1 x 1 number");
}

/// Members add: a member of A plus a member of B is a member of the sum with the stacked
/// coefficients; a batch adds per set.
void members_add(const std::string &b) {
    Rng rng(11);
    const MatZonotope A = test::random_set(rng, 3, 2, 0.5), B = test::random_set(rng, 3, 3, 0.5);
    const std::vector<double> ca = test::coefficients(rng, 2), cb = test::coefficients(rng, 3);
    std::vector<double> both = ca;
    both.insert(both.end(), cb.begin(), cb.end());
    check(close(test::member(A.plus(B), both), test::member(A, ca) + test::member(B, cb), 1e-12),
          b + ": the member with the joined coefficients is the sum");
    const MatZonotope batch = test::batch_of(A, A).plus(test::batch_of(A, A));
    check(test::same_set(test::batch_member(batch, 1), A.plus(A)), b + ": a batch");
}

/// Another size is refused; with the checks off a valid sum is the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    check(test::raises([&] { A.plus(MatZonotope(test::matrix({1, 2, 3}, 1, 3))); },
                       Error::dimension_mismatch),
          "plus: a matrix zonotope of another size");
    check(test::raises([&] { A.plus(test::matrix({1, 2, 3}, 1, 3)); }, Error::dimension_mismatch),
          "plus: a matrix of another size");
    const Tensor on = A.plus(A).center();
    ChecksScope off(false);
    check(close(A.plus(A).center(), on, 0), "plus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        sum_of_two_sets(b);
        sum_with_numbers(b);
        members_add(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
