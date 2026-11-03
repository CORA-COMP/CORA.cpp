// test_lean_interval_matrix_vertices - IntervalMatrix::vertices against the CORALean oracle
//
// The oracle lists the 2^(m n) corner matrices with repetitions; CORA.cpp's distinct ones, sorted
// as CORA's unique does, are those of the oracle after the same sorting (equal bounds repeat
// nothing here: the boxes have a positive radius).
//
// ORACLE: matrixSet/@intervalMatrix/vertices.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cstring>

using namespace cora;

namespace {

/// The matrices of the oracle (row-major values), sorted by the entries stacked column by column.
std::vector<std::vector<double>> aux_sorted(const lean::MatrixVertices &L, int64_t rows, int64_t cols) {
    std::vector<std::vector<double>> matrices;
    for (const lean::Tensor &V : L.V) matrices.push_back(V.gather().data());
    const auto stacked = [&](const std::vector<double> &M) {
        std::vector<double> v;
        for (int64_t j = 0; j < cols; ++j)
            for (int64_t i = 0; i < rows; ++i) v.push_back(M[i * cols + j]);
        return v;
    };
    std::sort(matrices.begin(), matrices.end(),
              [&](const auto &a, const auto &b) { return stacked(a) < stacked(b); });
    return matrices;
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("intervalMatrix.vertices", [](test::oracle::Random &rnd) {
        for (const auto [m, n] : {std::pair{2, 2}, std::pair{1, 3}, std::pair{3, 2}}) {
            const IntervalMatrix A(rnd.box_matrix(m, n));
            const lean::IntervalMatrix L{lean::Interval(A.interval())};
            const std::vector<std::vector<double>> theirs = aux_sorted(L.vertices(), m, n);
            const std::vector<double> mine = A.vertices()[0].data();
            bool same = mine.size() == theirs.size() * m * n;
            for (std::size_t k = 0; same && k < theirs.size(); ++k)
                same = std::memcmp(mine.data() + k * m * n, theirs[k].data(), m * n * sizeof(double)) == 0;
            test::check(same, "intervalMatrix.vertices: the same matrices as the oracle's");
        }
    });
    return test::finish("lean interval_matrix vertices");
}
