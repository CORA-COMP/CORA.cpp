// test_lean_interval_matrix_exponential_remainder - mat_exp::exponential_remainder against the
//
// CORALean oracle
//
// [-W, W] with W >= e^|A| - sum_{i <= eta} |A|^i / i!, e^|A| by scaling and squaring, is the
// oracle's bit for bit at nearest (MatExp.exponentialRemainder), with and without halvings; where
// no count below 128 halves |A| to norm 1/2 the oracle has none and CORA.cpp the unbounded box.
//
// ORACLE: matrixSet/@intervalMatrix/exponentialRemainder.m
// Authors:       Claude Opus 5.5
// Written:       20-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "cora/cont_set/interval/interval.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.exponentialRemainder", [](test::oracle::Random &rnd) {
        // small and large norms: no halvings and several
        for (const int n : {2, 3, 5})
            for (const double scale : {0.125, 1.0, 8.0}) {
                const Interval A = rnd.box_matrix(n, n) * scale;
                const lean::IntervalMatrix L{lean::Interval(A)};
                for (const int eta : {2, 4, 7}) {
                    const auto theirs = L.exponential_remainder(eta);
                    test::check(theirs.has_value(),
                                "exponentialRemainder: the oracle has halvings");
                    if (theirs)
                        test::oracle::check(mat_exp::exponential_remainder(A, eta), *theirs);
                }
            }
        // a norm no halving count below 128 brings to 1/2: unbounded, the oracle has none
        const Interval huge = rnd.box_matrix(2, 2) * 1e300;
        test::check(!lean::IntervalMatrix{lean::Interval(huge)}.exponential_remainder(2),
                    "exponentialRemainder: none without halvings");
        const std::vector<double> sup = mat_exp::exponential_remainder(huge, 2).sup().data();
        test::check(std::isinf(sup[0]) && std::isinf(sup[3]), "exponentialRemainder: unbounded");
    });
    return test::finish("lean interval_matrix exponential_remainder");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
