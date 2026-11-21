// test_lean_interval_matrix_norm - Interval::norm of an interval matrix against the CORALean oracle
//
// The norms one and inf are sums and maxima of the same entries. The Frobenius norm is a square
// root, which at dtype nearest the oracle takes as the machine does, as CORA.cpp on the lean
// backend: all three agree bit for bit.
//
// ORACLE: matrixSet/@intervalMatrix/norm.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   21-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.norm", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] : {std::pair{2, 2}, std::pair{3, 4}, std::pair{5, 1}}) {
            const Interval A = rnd.box_matrix(m, n);
            const lean::IntervalMatrix L{lean::Interval(A)};
            test::oracle::check("intervalMatrix.norm 1", A.norm(Tensor::Norm::one), L.norm("1"));
            test::oracle::check("intervalMatrix.norm inf", A.norm(Tensor::Norm::inf),
                                L.norm("inf"));
            test::oracle::check("intervalMatrix.norm fro", A.norm(Tensor::Norm::fro),
                                L.norm("fro"));
        }
    });
    return test::finish("lean interval_matrix norm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
