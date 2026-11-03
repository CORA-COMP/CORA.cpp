// test_lean_interval_matrix_exact_square - IntervalMatrix::exact_square against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/exactSquare.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.exactSquare", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 5}) {
            const IntervalMatrix A(rnd.box_matrix(n, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            test::oracle::check(A.exact_square(), L.exact_square());
        }
    });
    return test::finish("lean interval_matrix exact_square");
}
