// test_lean_interval_matrix_dependent_terms - IntervalMatrix::dependent_terms against the CORALean
// oracle
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
            // both results are rebuilt from center and radius, as CORA does (issue #53)
            for (const double r : {1.0, 0.5, 0.125}) {
                const auto [sq, H] = A.dependent_terms(r);
                const auto [lean_sq, lean_H] = L.dependent_terms(r);
                test::oracle::check_rebuilt(sq, lean_sq);
                test::oracle::check_rebuilt(H, lean_H);
            }
        }
    });
    return test::finish("lean interval_matrix dependent_terms");
}
