// test_interval_from_center_delta - the constructors of an interval matrix
//
// CORA: matrixSet/@intervalMatrix/intervalMatrix.m
// PROPERTY: matrixSet/@intervalMatrix/intervalMatrix.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
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

/// Center and delta give [C - |D|, C + |D|]; a center alone is a point; bounds are kept as they
/// are.
void the_three_constructors(const std::string &b) {
    const Tensor C = test::matrix({1, -2, 0, 3}, 2, 2), D = test::matrix({0.5, -1, 0, 2}, 2, 2);
    const Interval A = test::interval_matrix(C, D);
    check(close(A.infimum(), std::vector<double>{0.5, -3, 0, 1}), b + ": lower bound C - |D|");
    check(close(A.sup(), std::vector<double>{1.5, -1, 0, 5}), b + ": upper C + |D|");
    const Interval P = test::interval_matrix(C);
    check(close(P.infimum(), C) && close(P.sup(), C), b + ": a point matrix");
    const Interval E = test::example();
    check(E.inf().size(-2) == 2 && E.inf().size(-1) == 2, b + ": the size");
    check(close(test::interval_matrix(E).infimum(), E.infimum()), b + ": from an interval");
}

/// A set built from the center and the radius of a random set is that set up to rounding, and it
/// holds its center.
void rebuilt_from_center_and_radius(const std::string &b) {
    Rng rng(5);
    const Interval A = test::random_matrix(rng, 3, 1.0);
    const Interval B = test::interval_matrix(A.center(), A.rad());
    check(close(B.infimum(), A.infimum(), 1e-12) && close(B.sup(), A.sup(), 1e-12),
          b + ": the center and the radius rebuild the set");
    check(B.contains(A.center(), 0.0), b + ": it contains its center");
}

/// A center and a delta of different shapes are refused; with the checks off a valid one is the
/// same.
void checks_and_their_switch() {
    const Tensor C = test::matrix({1, 2, 3, 4}, 2, 2), D = test::matrix({1, 2, 3}, 1, 3);
    check(test::raises([&] { test::interval_matrix(C, D); }, Error::wrong_input_in_constructor),
          "Interval: a delta of another shape");
    const Tensor on = test::interval_matrix(C, C).infimum();
    ChecksScope off(false);
    check(close(test::interval_matrix(C, C).infimum(), on, 0),
          "Interval: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        the_three_constructors(b);
        rebuilt_from_center_and_radius(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix constructor");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
