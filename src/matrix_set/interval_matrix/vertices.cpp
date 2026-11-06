// vertices - the vertices of an interval matrix, as CORA's intervalMatrix.vertices
//
// Syntax:   std::vector<Tensor> V = A.vertices();
// Inputs:   -;  at most 20 entries (2^20 vertices), else Error::not_supported
// Outputs:  V - one tensor per batch member (the batch flattened, row-major), V[b] of shape
//           (K, rows, cols): the K distinct matrices with every entry at an end of its interval,
//           sorted by the entries stacked column by column (CORA's unique(V', 'rows'))
// See also: representsa, Interval::vertices
//
// CORA: matrixSet/@intervalMatrix/vertices.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "matrix_set/interval_matrix/private/priv_vec.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The most entries whose 2^n corners are listed.
constexpr int64_t kMaxEntries = 20;

/// The corners of the stacked box (..., n, 1) as (..., 2^n, n): row c takes the upper end in
/// coordinate j where bit n - 1 - j of c is set.
Tensor aux_corners(const Interval &v, int64_t n) {
    const int64_t corners = int64_t(1) << n;
    std::vector<double> bits(corners * n);
    for (int64_t c = 0; c < corners; ++c)
        for (int64_t j = 0; j < n; ++j) bits[c * n + j] = double((c >> (n - 1 - j)) & 1);
    const Tensor upper = Tensor::like(v.inf(), bits, {corners, n});
    const Tensor lo = v.inf().squeeze(-1).expand_dims(-2), hi = v.sup().squeeze(-1).expand_dims(-2);
    return Tensor::where(upper, hi, lo);
}

/// The rows of the (K, n) host values without repetition, in ascending lexicographic order: the
/// index of the first of each.
std::vector<int64_t> aux_unique_rows(const double *values, int64_t K, int64_t n) {
    std::vector<int64_t> order(K);
    std::iota(order.begin(), order.end(), 0);
    const auto row_less = [&](int64_t a, int64_t b) {
        return std::lexicographical_compare(values + a * n, values + (a + 1) * n, values + b * n,
                                            values + (b + 1) * n);
    };
    std::stable_sort(order.begin(), order.end(), row_less);
    std::vector<int64_t> keep;
    for (const int64_t c : order)
        if (keep.empty() || row_less(keep.back(), c)) keep.push_back(c);
    return keep;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> IntervalMatrix::vertices() const {
    detail::Entry entry;
    const int64_t n = rows() * cols();
    CORA_CHECK(n <= kMaxEntries, Error::not_supported, [&] {
        return "vertices: " + std::to_string(n) + " entries give 2^" + std::to_string(n) +
               " vertices, at most " + std::to_string(kMaxEntries) + " entries are supported";
    });
    const Tensor corners = aux_corners(priv_vec(int_), n); // (..., K, n)
    const int64_t K = corners.size(-2);
    int64_t members = 1;
    for (int64_t d = 0; d + 2 < int_.inf().ndim(); ++d) members *= int_.inf().size(int(d));
    const Tensor flat = corners.reshape({members, K, n});
    const std::vector<double> values = flat.data();
    std::vector<Tensor> out;
    for (int64_t b = 0; b < members; ++b) {
        const std::vector<int64_t> keep = aux_unique_rows(values.data() + b * K * n, K, n);
        const Tensor rows_kept = flat.slice(0, b, b + 1).squeeze(0).take(keep, 0); // (K', n)
        const int64_t count = int64_t(keep.size());
        out.push_back(rows_kept.reshape({count, cols(), rows()}).matrix_transpose());
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
