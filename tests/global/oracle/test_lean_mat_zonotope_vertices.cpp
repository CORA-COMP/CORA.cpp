// test_lean_mat_zonotope_vertices - MatZonotope::vertices against the CORALean oracle
//
// The oracle lists the 2^h corner matrices with repeats; CORA.cpp's list has each once, in the
// ascending order of the stacked matrices (MATLAB's unique). The oracle's list, sorted and without
// repeats, is the same bits. The oracle adds
// the signed generators from the last to the first, then the center.
//
// ORACLE: matrixSet/@matZonotope/vertices.m
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/mat_zonotope.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "oracle_testing.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The matrices of the oracle (row-major values), sorted by the entries stacked column by column
/// and without repeats.
std::vector<std::vector<double>> aux_sorted(const lean::MatrixVertices &L, int64_t rows,
                                            int64_t cols) {
    std::vector<std::vector<double>> matrices;
    matrices.reserve(L.V.size());
    for (const lean::Tensor &V : L.V) matrices.push_back(V.gather().data());
    const auto stacked = [&](const std::vector<double> &M) {
        std::vector<double> v;
        for (int64_t j = 0; j < cols; ++j)
            for (int64_t i = 0; i < rows; ++i) v.push_back(M[i * cols + j]);
        return v;
    };
    std::sort(matrices.begin(), matrices.end(),
              [&](const auto &a, const auto &b) { return stacked(a) < stacked(b); });
    matrices.erase(std::unique(matrices.begin(), matrices.end()), matrices.end());
    return matrices;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    oracle::run("matZonotope.vertices", [](oracle::Random &rnd) {
        for (const auto &[m, n, h] :
             {std::tuple{2, 2, 3}, std::tuple{1, 3, 2}, std::tuple{3, 2, 4}}) {
            const MatZonotope A = rnd.mat_zonotope(m, n, h);
            const lean::MatrixVertices reply = lean::MatZonotope(A).vertices();
            const std::vector<std::vector<double>> theirs = aux_sorted(reply, m, n);
            // both lists as one row of entries per corner matrix
            const std::vector<double> mine = A.vertices()[0].data();
            std::vector<double> stacked;
            for (const std::vector<double> &corner : theirs)
                stacked.insert(stacked.end(), corner.begin(), corner.end());
            const int64_t entries = int64_t(m) * n;
            oracle::check(Tensor::from_data(mine, {int64_t(mine.size()) / entries, entries}),
                          Tensor::from_data(stacked, {int64_t(theirs.size()), entries}));
        }
    });
    return test::finish("lean mat zonotope vertices");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
