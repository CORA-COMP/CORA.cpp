// test_interval_interval_matrix_bounds - Interval interval: the bounds as an Interval
//
// CORA: matrixSet/@intervalMatrix/interval.m
// PROPERTY: matrixSet/@intervalMatrix/interval.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The bounds of the example, and C +- |D| for the constructor of a center and a delta.
void bounds_of_the_set(const std::string &b) {
    const Interval I = test::example();
    check(close(I.inf(), std::vector<double>{1, -1, 0, 2}) &&
              close(I.sup(), std::vector<double>{2, 1, 3, 2}),
          b + ": the example");
    const Interval A =
        test::interval_matrix(test::matrix({1, -2}, 1, 2), test::matrix({0.5, -1}, 1, 2));
    check(close(A.inf(), std::vector<double>{0.5, -3}) &&
              close(A.sup(), std::vector<double>{1.5, -1}),
          b + ": C +- |D|");
}

/// A set made of an interval has that interval back, and the members lie in its bounds.
void interval_round_trip(const std::string &b) {
    Rng rng(24);
    const Interval A = test::random_matrix(rng, 3, 1.0);
    const Interval B = test::interval_matrix(A);
    check(close(B.inf(), A.inf(), 0) && close(B.sup(), A.sup(), 0), b + ": the same bounds");
    bool inside = true;
    for (int trial = 0; trial < 20; ++trial) {
        const Tensor M = test::member(A, rng);
        inside = inside && A.inf().less_equal(M).min().item() != 0 &&
                 M.less_equal(A.sup()).min().item() != 0;
    }
    check(inside, b + ": members are inside the bounds");
}

/// interval has no input to refuse; with the checks off the result is the same.
void checks_and_their_switch() {
    const Tensor on = test::example().inf();
    ChecksScope off(false);
    check(close(test::example().inf(), on, 0), "interval: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        bounds_of_the_set(b);
        interval_round_trip(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
