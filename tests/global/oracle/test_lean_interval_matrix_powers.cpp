// test_lean_interval_matrix_powers - IntervalMatrix::powers against the CORALean oracle
//
// The power of order initial_order + k is the oracle's powers at offset k from the initial power.
//
// ORACLE: matrixSet/@intervalMatrix/powers.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.powers", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 4}) {
            const IntervalMatrix A(rnd.box_matrix(n, n)), X(rnd.box_matrix(n, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            for (const int k : {0, 1, 2, 4}) {
                test::oracle::check(A.powers(k + 1).back(),
                                    L.powers(lean::Interval(A.interval()), k));
                test::oracle::check(A.powers(k + 3, 3, X).back(),
                                    L.powers(lean::Interval(X.interval()), k));
            }
        }
    });
    return test::finish("lean interval_matrix powers");
}
