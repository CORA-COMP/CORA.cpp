// test_lean_interval_matrix_dependent_terms - IntervalMatrix::dependent_terms against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/dependentTerms.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.dependentTerms", [](test::oracle::Random &rnd) {
        for (const int n : {2, 3, 5}) {
            const IntervalMatrix A(rnd.box_matrix(n, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            for (const double r : {1.0, 0.5, 0.125})
                test::oracle::check(A.dependent_terms(r), L.dependent_terms(r));
        }
    });
    return test::finish("lean interval_matrix dependent_terms");
}
