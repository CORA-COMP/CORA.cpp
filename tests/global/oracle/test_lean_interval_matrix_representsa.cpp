// test_lean_interval_matrix_representsa - IntervalMatrix::representsa against the CORALean oracle
//
// The oracle decides emptySet, point and origin of the stacked entries, with tol 0 only; a true
// is certain. At dtype nearest the oracle answers emptySet true for a box with inf = sup (its
// strict comparison is not strict there), so emptySet is compared on boxes of positive radius only.
//
// ORACLE: matrixSet/@intervalMatrix/representsa.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.representsa", [](test::oracle::Random &rnd) {
        for (const auto [m, n] : {std::pair{2, 2}, std::pair{3, 1}, std::pair{1, 4}}) {
            const Tensor M = rnd.matrix(m, n);
            const Tensor thin = M.full_like(M.shape(), 1e-3);
            const auto compare = [&](const IntervalMatrix &A, bool positive_radius) {
                const lean::IntervalMatrix L{lean::Interval(A.interval())};
                for (const char *type : {"emptySet", "point", "origin"})
                    if (positive_radius || std::string(type) != "emptySet")
                        test::oracle::check(A.representsa(type, 0.0), L.representsa(type, 0.0));
            };
            compare(IntervalMatrix(rnd.box_matrix(m, n)), true);
            compare(IntervalMatrix(Interval(M - thin, M + thin)), true);
            compare(IntervalMatrix(M), false);
            compare(IntervalMatrix(M * 0.0), false);
        }
    });
    return test::finish("lean interval_matrix representsa");
}
