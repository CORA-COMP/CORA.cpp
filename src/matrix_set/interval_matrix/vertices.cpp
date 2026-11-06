// vertices - the vertices of an interval matrix, as CORA's intervalMatrix.vertices
//
// The batch is flattened row-major. V[b] holds the K distinct matrices with every entry at an end
// of its interval, sorted by the entries stacked column by column (CORA's unique(V', 'rows')).
//
// Syntax:   std::vector<Tensor> V = A.vertices();
// Inputs:   -   (at most 20 entries, 2^20 vertices, else Error::not_supported)
// Outputs:  V - one tensor per batch member, V[b] of shape (K, rows, cols)
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
#include "matrix_set/private/priv_vertex_matrices.h"


// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


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

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> IntervalMatrix::vertices() const {
    detail::Entry entry;
    const int64_t n = rows() * cols();
    CORA_CHECK(n <= kMaxCornerCoordinates, Error::not_supported, [&] {
        return "vertices: " + std::to_string(n) + " entries give 2^" + std::to_string(n) +
               " vertices, at most " + std::to_string(kMaxCornerCoordinates) +
               " entries are supported";
    });
    return priv_vertex_matrices(aux_corners(priv_vec(int_), n), rows(), cols());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
