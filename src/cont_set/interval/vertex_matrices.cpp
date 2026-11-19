// vertex_matrices - the vertices of an interval matrix, as CORA's intervalMatrix.vertices
//
// The batch is flattened row-major. V[b] holds the K distinct matrices with every entry at an end
// of its interval, sorted by the entries stacked column by column (CORA's unique(V', 'rows')).
//
// Syntax:   std::vector<Tensor> V = A.vertex_matrices();
// Inputs:   -   (at most 20 entries, 2^20 vertices, else Error::not_supported)
// Outputs:  V - one tensor per batch member, V[b] of shape (K, rows, cols)
// See also: representsa, vertices (the vertices of a box of columns)
//
// CORA: matrixSet/@intervalMatrix/vertices.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "global/vertex_matrices.h"
#include "cont_set/interval/private/priv_vec.h"


// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// The corners of the stacked box (..., n) as (..., 2^n, n): row c takes the upper end in
/// coordinate j where bit n - 1 - j of c is set.
Tensor aux_corners(const Interval &v, int64_t n) {
    const int64_t corners = int64_t(1) << n;
    std::vector<double> bits(corners * n);
    for (int64_t c = 0; c < corners; ++c)
        for (int64_t j = 0; j < n; ++j) bits[c * n + j] = double((c >> (n - 1 - j)) & 1);
    const Tensor upper = Tensor::like(v.inf(), bits, {corners, n});
    const Tensor lo = v.inf().expand_dims(-2), hi = v.sup().expand_dims(-2);
    return Tensor::where(upper, hi, lo);
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Tensor> Interval::vertex_matrices() const {
    detail::Entry entry;
    const int64_t rows = dim(1), cols = dim(2), n = rows * cols;
    CORA_CHECK(n <= kMaxCornerCoordinates, Error::not_supported, [&] {
        return "vertex_matrices: " + std::to_string(n) + " entries give 2^" + std::to_string(n) +
               " vertices, at most " + std::to_string(kMaxCornerCoordinates) +
               " entries are supported";
    });
    return cora::vertex_matrices(aux_corners(priv_vec(*this), n), rows, cols);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
