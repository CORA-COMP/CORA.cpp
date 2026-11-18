// test_lean_interval_matrix_mat_zonotope - Interval::mat_zonotope against the CORALean oracle
//
// Bit for bit: the center matrix and one generator per entry (CORALean numbers them by rows).
//
// ORACLE: matrixSet/@intervalMatrix/matZonotope.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "cora/global/oracle/interval_matrix.h"
#include "cora/global/oracle/mat_zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The set with its generators in CORALean's order: the entry (i, j) is generator i m + j (row
/// major), where CORA numbers them j n + i (column major).
Zonotope row_major(const Zonotope &A) {
    const int64_t n = A.center().size(-2), m = A.center().size(-1), h = n * m;
    const std::vector<double> in = A.generators().data();
    std::vector<double> out(in.size());
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j)
            for (int64_t e = 0; e < h; ++e)
                out[std::size_t(((i * m + j) * n + e / m) * m + e % m)] =
                    in[std::size_t(((j * n + i) * n + e / m) * m + e % m)];
    return Zonotope::matrix(A.center(), Tensor::like(A.center(), out, {h, n, m}));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("intervalMatrix.matZonotope", [](oracle::Random &rnd) {
        for (const auto &[m, n] : {std::pair{2, 2}, std::pair{1, 3}, std::pair{3, 2}}) {
            const Interval A = rnd.box_matrix(m, n);
            const lean::MatZonotope theirs = lean::IntervalMatrix{lean::Interval(A)}.mat_zonotope();
            oracle::check(row_major(Zonotope::from_interval_matrix(A)), theirs);
        }
    });
    return test::finish("lean interval matrix mat_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
