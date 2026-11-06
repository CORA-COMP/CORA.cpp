// test_mat_zonotope_contains - MatZonotope contains: whether matrices are in the set
//
// CORA: matrixSet/@matZonotope/contains.m
// PROPERTY: matrixSet/@matZonotope/contains.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "mat_zonotope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// C + 0.5 G1 - 0.3 G2 of the example is inside; the corner C + G1 + G2 is on the boundary, and
/// C + 2 G1 and C + (3, 0; 0, 0) are outside. p matrices are inside only if each one is.
void example_members(const std::string &b) {
    const MatZonotope A = test::example();
    const Tensor inside = test::member(A, {0.5, -0.3}), corner = test::member(A, {1, 1});
    check(A.contains(inside, "exact", 1e-9), b + ": an interior member");
    check(A.contains(corner, "exact", 1e-9), b + ": a corner");
    check(!A.contains(test::member(A, {2, 0}), "exact", 1e-9), b + ": outside along a generator");
    check(!A.contains(A.center() + test::matrix({3, 0, 0, 0}, 2, 2), "exact", 1e-9),
          b + ": outside the span of the generators");
    check(A.contains(Tensor::stack({inside, corner}, 0), "exact", 1e-9), b + ": two members");
    check(!A.contains(Tensor::stack({inside, test::member(A, {2, 0})}, 0), "exact", 1e-9),
          b + ": one of two outside");
    check(A.contains(inside, "approx", 1e-9), b + ": the type approx");
}

/// The tolerance admits a matrix that lies slightly outside; the default is the repository's.
void tolerance(const std::string &b) {
    const MatZonotope A = test::example();
    const Tensor near = test::member(A, {1.0 + 1e-8, 0.0});
    check(!A.contains(near, "exact", 1e-12), b + ": outside by 1e-8 without slack");
    check(A.contains(near, "exact", 1e-6), b + ": inside with the tolerance 1e-6");
    check(A.contains(A.center()), b + ": the default tolerance");
}

/// Random members of a random set are inside.
void random_members(const std::string &b) {
    Rng rng(9);
    const MatZonotope A = test::random_set(rng, 3, 5, 0.4);
    bool all = true;
    for (int trial = 0; trial < 20; ++trial)
        all = all && A.contains(test::member(A, test::coefficients(rng, 5)), "exact", 1e-9);
    check(all, b + ": members of a random set");
}

/// A wrong type, a matrix of another size and a batch of sets are refused; checks off: the same.
void checks_and_their_switch() {
    const MatZonotope A = test::example();
    const Tensor M = A.center();
    check(test::raises([&] { A.contains(M, "sampling"); }, Error::wrong_value),
          "contains: the type");
    check(test::raises([&] { A.contains(M, "exact", -1.0); }, Error::wrong_value), "contains: tol");
    check(
        test::raises([&] { A.contains(test::matrix({1, 2, 3}, 1, 3)); }, Error::dimension_mismatch),
        "contains: the size of the matrix");
    check(test::raises([&] { test::batch_of(A, A).contains(M); }, Error::not_supported),
          "contains: a batch of sets");
    const bool on = A.contains(M);
    ChecksScope off(false);
    check(A.contains(M) == on, "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        example_members(b);
        tolerance(b);
        random_members(b);
    });
    checks_and_their_switch();
    return test::finish("mat_zonotope contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
