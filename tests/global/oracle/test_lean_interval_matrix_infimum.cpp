// test_lean_interval_matrix_infimum - IntervalMatrix::infimum against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/infimum.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.infimum", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] :
             {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}, std::pair{4, 4}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            test::oracle::check(A.infimum(), L.infimum());
        }
    });
    return test::finish("lean interval_matrix infimum");
}
