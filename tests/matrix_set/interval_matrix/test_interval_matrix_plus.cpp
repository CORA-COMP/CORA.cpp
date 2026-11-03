// test_interval_matrix_plus - IntervalMatrix plus: the sum with an interval matrix, a matrix, a number
//
// CORA: matrixSet/@intervalMatrix/plus.m
// PROPERTY: matrixSet/@intervalMatrix/plus.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// A + A adds the bounds; a matrix and a number are points; the sum commutes.
void sums_of_the_example(const std::string &b) {
    const IntervalMatrix A = test::example();
    const Interval AA = A.plus(A).interval();
    check(close(AA.inf(), std::vector<double>{2, -2, 0, 4}) &&
              close(AA.sup(), std::vector<double>{4, 2, 6, 4}),
          b + ": A + A");
    const Tensor M = test::matrix({1, 2, 3, 4}, 2, 2);
    const Interval AM = (A + M).interval(), MA = (M + A).interval();
    check(close(AM.inf(), std::vector<double>{2, 1, 3, 6}) &&
              close(AM.sup(), std::vector<double>{3, 3, 6, 6}),
          b + ": A + M");
    check(close(MA.inf(), AM.inf(), 0) && close(MA.sup(), AM.sup(), 0), b + ": M + A is A + M");
    const Interval A_plus_s = (A + 1.0).interval(), s_plus_A = (1.0 + A).interval();
    check(close(A_plus_s.inf(), std::vector<double>{2, 0, 1, 3}) &&
              close(A_plus_s.sup(), std::vector<double>{3, 2, 4, 3}),
          b + ": A + 1");
    check(close(s_plus_A.inf(), A_plus_s.inf(), 0), b + ": 1 + A is A + 1");
}

/// Sums of members are members of the sum, and the radii add; a batch is per member.
void members_add_up(const std::string &b) {
    Rng rng(12);
    const IntervalMatrix A = test::random_matrix(rng, 3, 1.0), B = test::random_matrix(rng, 3, 0.5);
    const IntervalMatrix S = A + B;
    bool inside = true;
    for (int trial = 0; trial < 30; ++trial)
        inside = inside && S.contains(test::member(A, rng) + test::member(B, rng), "exact", 1e-12);
    check(inside, b + ": members add up");
    check(close(S.rad(), A.rad() + B.rad(), 1e-12), b + ": the radii add");
    const IntervalMatrix batch(Interval::stack({A.interval(), B.interval()}));
    const Tensor sum = (batch + batch).infimum();
    check(close(sum.slice(0, 0, 1).reshape({3, 3}), A.infimum() * 2.0, 1e-12), b + ": batch");
}

/// A matrix of another shape is refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    const IntervalMatrix A = test::example();
    check(test::raises([&] { A.plus(IntervalMatrix(test::matrix({1, 2, 3}, 1, 3))); },
                       Error::dimension_mismatch),
          "plus: an interval matrix of another shape");
    check(test::raises([&] { A.plus(test::matrix({1, 2, 3}, 1, 3)); }, Error::dimension_mismatch),
          "plus: a matrix of another shape");
    const Tensor on = (A + A).infimum();
    ChecksScope off(false);
    check(close((A + A).infimum(), on, 0), "plus: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        sums_of_the_example(b);
        members_add_up(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix plus");
}
