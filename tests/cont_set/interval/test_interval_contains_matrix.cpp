// test_interval_contains_matrix - Interval contains: whether a matrix is in the set
//
// CORA: matrixSet/@intervalMatrix/contains.m
// PROPERTY: matrixSet/@intervalMatrix/contains.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Inside, on the boundary and outside of the example; the tolerance (absolute or relative to the
/// bound, as within_tol) moves the boundary.
void containment_of_the_example(const std::string &b) {
    const Interval A = test::example();
    check(A.contains(A.center(), "exact", 0.0).all(), b + ": the center");
    check(A.contains(A.infimum(), "exact", 0.0).all(), b + ": the lower bound");
    check(A.contains(A.sup(), "exact", 0.0).all(), b + ": the upper bound");
    const Tensor out = test::matrix({1, -1, 3.5, 2}, 2, 2);
    check(!A.contains(out, "exact", 0.0).all(), b + ": one entry above its bound");
    check(A.contains(out, "exact", 0.2).all(),
          b + ": the tolerance 0.2 admits it (0.2 of the bound 3)");
    check(!A.contains(out, "exact", 0.05).all(), b + ": the tolerance 0.05 does not");
    check(A.contains(A.center(), "approx", 0.0).all(), b + ": type approx gives the same answer");
    check(!A.contains(out, "approx", 0.0).all(), b + ": type approx, outside");
}

/// Members of random sets are in, a shifted copy is out; a batch is in if all members are.
void members_are_in(const std::string &b) {
    Rng rng(8);
    const Interval A = test::random_matrix(rng, 3, 1.0);
    bool all_in = true, none_out = true;
    for (int trial = 0; trial < 30; ++trial) {
        const Tensor M = test::member(A, rng);
        all_in = all_in && A.contains(M, "exact", 0.0).all();
        none_out = none_out && !A.contains(M + 5.0, "exact", 0.0).all();
    }
    check(all_in && none_out, b + ": members in, shifted members out");
    const Interval T = test::random_matrix(rng, 2, 1.0);
    const Interval batch = Interval::stack({T, test::example()});
    const Tensor both = Tensor::stack({T.center(), test::example().center()});
    check(batch.contains(both, "exact", 0.0).all(), b + ": a batch of centers");
    const Tensor shifted = Tensor::stack({T.center(), test::example().center() + 9.0});
    check(!batch.contains(shifted, "exact", 0.0).all(), b + ": one batch member outside");
}

/// A wrong size, a wrong type and a negative tolerance are refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval A = test::example();
    check(test::raises([&] { A.contains(test::matrix({1, 2, 3}, 1, 3)).all(); },
                       Error::dimension_mismatch),
          "contains: a matrix of another size");
    check(test::raises([&] { A.contains(A.center(), "other").all(); }, Error::wrong_value),
          "contains: an unknown type");
    check(test::raises([&] { A.contains(A.center(), "exact", -1.0).all(); }, Error::wrong_value),
          "contains: a negative tolerance");
    ChecksScope off(false);
    check(A.contains(A.center(), "exact", 0.0).all(), "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        containment_of_the_example(b);
        members_are_in(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
