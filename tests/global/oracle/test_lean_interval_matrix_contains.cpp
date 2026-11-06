// test_lean_interval_matrix_contains - IntervalMatrix::contains against the CORALean oracle
//
// ORACLE: matrixSet/@intervalMatrix/contains.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.contains", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] : {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            // the center and the bounds are in, a shifted matrix is out; tol 0 and a tolerance
            const Tensor shifted = A.center() + 5.0;
            for (const Tensor &M : {A.center(), A.infimum(), A.interval().sup(), shifted})
                for (const double tol : {0.0, 0.0009765625}) {
                    const lean::Tensor M_lean = lean::Tensor::from(M);
                    test::oracle::check(A.contains(M, "exact", tol), L.contains(M_lean, tol));
                }
        }
    });
    return test::finish("lean interval_matrix contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
