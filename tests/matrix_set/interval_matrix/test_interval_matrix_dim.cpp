// test_interval_matrix_dim - IntervalMatrix dim: the number of rows and columns
//
// CORA: matrixSet/@intervalMatrix/dim.m
// PROPERTY: matrixSet/@intervalMatrix/dim.m

#include "interval_matrix_testing.h"

using namespace cora;
using test::check;

namespace {

/// dim is {rows, cols}, and dim(rc) one of them; a batch has the size of its members.
void dimensions_of_matrices(const std::string &b) {
    Rng rng(22);
    const IntervalMatrix wide(test::matrix({1, 2, 3}, 1, 3));
    check(test::example().dim() == std::array<int64_t, 2>{2, 2}, b + ": 2 x 2");
    check(wide.dim() == std::array<int64_t, 2>{1, 3}, b + ": 1 x 3");
    check(wide.dim(1) == 1 && wide.dim(2) == 3, b + ": dim(rc)");
    for (const int64_t rows : {1, 4, 7}) {
        std::vector<double> c(rows * 2, 1.0);
        const IntervalMatrix A(test::matrix(c, rows, 2), test::matrix(c, rows, 2));
        check(A.dim() == std::array<int64_t, 2>{rows, 2} && A.rows() == rows, b + ": random sizes");
    }
    const IntervalMatrix T = test::random_matrix(rng, 2, 1.0);
    const IntervalMatrix batch(Interval::stack({T.interval(), test::example().interval()}));
    check(batch.dim() == std::array<int64_t, 2>{2, 2}, b + ": a batch");
}

/// rc other than 1 and 2 is refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    check(test::raises([&] { test::example().dim(0); }, Error::wrong_value), "dim: rc = 0");
    check(test::raises([&] { test::example().dim(3); }, Error::wrong_value), "dim: rc = 3");
    ChecksScope off(false);
    check(test::example().dim(2) == 2, "dim: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend_and_lean(dimensions_of_matrices);
    checks_and_their_switch();
    return test::finish("interval_matrix dim");
}
