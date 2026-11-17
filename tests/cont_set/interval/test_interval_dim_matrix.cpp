// test_interval_dim_matrix - Interval dim(rc): the number of rows and columns of an interval matrix
//
// CORA: matrixSet/@intervalMatrix/dim.m
// PROPERTY: matrixSet/@intervalMatrix/dim.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// dim(1) is the rows and dim(2) the columns, dim() the rows; a batch has the size of its members.
void dimensions_of_matrices(const std::string &b) {
    Rng rng(22);
    const Interval wide = test::interval_matrix(test::matrix({1, 2, 3}, 1, 3));
    check(test::example().dim(1) == 2 && test::example().dim(2) == 2, b + ": 2 x 2");
    check(wide.dim(1) == 1 && wide.dim(2) == 3 && wide.dim() == 1, b + ": 1 x 3");
    for (const int64_t rows : {1, 4, 7}) {
        std::vector<double> c(rows * 2, 1.0);
        const Interval A =
            test::interval_matrix(test::matrix(c, rows, 2), test::matrix(c, rows, 2));
        check(A.dim(1) == rows && A.dim(2) == 2, b + ": random sizes");
    }
    const Interval T = test::random_matrix(rng, 2, 1.0);
    const Interval batch = Interval::stack({T, test::example()});
    check(batch.dim(1) == 2 && batch.dim(2) == 2, b + ": a batch");
}

/// rc other than 1 and 2 is refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    check(test::raises([&] { test::example().dim(0); }, Error::wrong_value), "dim: rc = 0");
    check(test::raises([&] { test::example().dim(3); }, Error::wrong_value), "dim: rc = 3");
    ChecksScope off(false);
    check(test::example().dim(2) == 2, "dim: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend_and_lean(dimensions_of_matrices);
    checks_and_their_switch();
    return test::finish("interval dim matrix");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
