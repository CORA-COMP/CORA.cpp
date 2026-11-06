// test_interval_matrix_exact_square - IntervalMatrix exact_square: the range of M * M
//
// CORA: matrixSet/@intervalMatrix/exactSquare.m
// PROPERTY: matrixSet/@intervalMatrix/exactSquare.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// By hand: (1,1) is [1, 4] + [-1, 1] [0, 3] = [-2, 7], (1,2) is [-1, 1] ([1, 2] + [2, 2]) = [-4,
/// 4], (2,1) is [0, 3] [3, 4] = [0, 12] and (2,2) is 4 + [-3, 3] = [1, 7].
void square_of_the_example(const std::string &b) {
    const Interval S = test::example().exact_square().interval();
    check(close(S.inf(), std::vector<double>{-2, -4, 0, 1}), b + ": lower bounds");
    check(close(S.sup(), std::vector<double>{7, 4, 12, 7}), b + ": upper bounds");
    const Tensor M = test::matrix({1, 2, 3, 4}, 2, 2);
    const Interval P = IntervalMatrix(M).exact_square().interval();
    check(close(P.inf(), std::vector<double>{7, 10, 15, 22}) && close(P.sup(), P.inf()),
          b + ": a point matrix is its square");
    // a diagonal entry that straddles 0 squares to [0, ...], one that touches 0 does as well
    const IntervalMatrix D(
        Interval(test::matrix({-2, 0}, 2, 1).diag(), test::matrix({1, 3}, 2, 1).diag()));
    check(close(D.exact_square().interval().inf(), std::vector<double>{0, 0, 0, 0}),
          b + ": 0 inside");
    check(close(D.exact_square().interval().sup(), std::vector<double>{4, 0, 0, 9}),
          b + ": its end");
}

/// The squares of members of random sets are in the result.
void squares_of_members_are_inside(const std::string &b) {
    Rng rng(9);
    for (const int n : {2, 3, 4}) {
        const IntervalMatrix A = test::random_matrix(rng, n, 0.8);
        const Interval S = A.exact_square().interval();
        bool inside = true;
        for (int trial = 0; trial < 40; ++trial) {
            const Tensor M = test::member(A, rng), M2 = M.matmul(M);
            inside = inside && S.inf().less_equal(M2 + 1e-12).min().item() != 0 &&
                     M2.less_equal(S.sup() + 1e-12).min().item() != 0;
        }
        check(inside, b + ": M * M in the range, n = " + std::to_string(n));
    }
}

/// A matrix that is not square is refused; with the checks off a valid one is the same.
void checks_and_their_switch() {
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3));
    check(test::raises([&] { wide.exact_square(); }, Error::dimension_mismatch),
          "exact_square: a matrix that is not square");
    const Tensor on = test::example().exact_square().interval().sup();
    ChecksScope off(false);
    check(close(test::example().exact_square().interval().sup(), on, 0),
          "exact_square: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean([](const std::string &b) {
        square_of_the_example(b);
        squares_of_members_are_inside(b);
    });
    checks_and_their_switch();
    return test::finish("interval_matrix exact_square");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
