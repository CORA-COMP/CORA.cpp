// test_lean_interval_matrix_representsa - Interval::representsa against the CORALean oracle
//
// The oracle decides emptySet, point and origin of the stacked entries, with tol 0 only; a true
// is certain. At dtype nearest the oracle answers emptySet true for a box with inf = sup (its
// strict comparison is not strict there), so emptySet is compared on boxes of positive radius only.
//
// ORACLE: matrixSet/@intervalMatrix/representsa.m
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/interval_matrix_testing.h"
#include "cora/cont_set/interval/interval.h"
#include "oracle_testing.h"

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.representsa", [](test::oracle::Random &rnd) {
        for (const auto &[m, n] : {std::pair{2, 2}, std::pair{3, 1}, std::pair{1, 4}}) {
            const Tensor M = rnd.matrix(m, n);
            const Tensor thin = M.full_like(M.shape(), 1e-3);
            const auto compare = [&](const Interval &A, bool positive_radius) {
                const lean::IntervalMatrix L{lean::Interval(A)};
                for (const char *type : {"emptySet", "point", "origin"})
                    if (positive_radius || std::string(type) != "emptySet")
                        test::oracle::check(A.representsa(type, 0.0), L.representsa(type, 0.0));
            };
            compare(test::interval_matrix(rnd.box_matrix(m, n)), true);
            compare(Interval::matrix(M - thin, M + thin), true);
            compare(test::interval_matrix(M), false);
            compare(test::interval_matrix(M * 0.0), false);
        }
    });
    return test::finish("lean interval_matrix representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
