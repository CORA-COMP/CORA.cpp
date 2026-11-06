// volume - the volume of a zonotope, exactly or by an upper bound
//
// "exact": 2^n times the sum of |det G_S| over the n-subsets S of the generators, taken in
// blocks; "alamo": 2^n sqrt(det(G G')), a lower bound; "reduce" is an order reduction by 'pca'.
//
// Syntax:   v = Z.volume();   v = Z.volume(method);   v = Z.volume("reduce", order);
// Inputs:   method - "exact" (default) or "alamo"; "reduce" is not supported
//           order - the order of "reduce"
// Outputs:  v - (..., 1, 1); 0 if the generators do not span the space (rank below n)
// See also: is_full_dim, interval
//
// CORA: contSet/@zonotope/volume_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   06-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the rank of G (..., n, m) is below n, as (..., 1, 1): the smallest singular value is at
/// most max(n, m) eps (largest), as MATLAB's rank (n <= m).
Tensor aux_rank_deficient(const Tensor &G) {
    const int64_t n = G.shape()[G.ndim() - 2], m = G.shape().back();
    const Tensor values = G.detach().svd()[1]; // (..., n, 1), descending
    const double scale = double(std::max(n, m)) * machine_eps(G.dtype());
    const Tensor bound = values.slice(-2, 0, 1) * scale;
    return values.slice(-2, n - 1, n).less_equal(bound);
}

/// The sum of |det G_S| over the n-subsets S of the columns of G (..., n, m), as (..., 1, 1), a
/// block of subsets at a time (CORA's batch of 1e7).
Tensor aux_sum_of_determinants(const Tensor &G) {
    const int64_t n = G.shape()[G.ndim() - 2], m = G.shape().back(), batch = G.ndim() - 2;
    const int64_t members = std::max<int64_t>(1, G.numel() / std::max<int64_t>(1, n * m));
    const int64_t block = std::max<int64_t>(1, numerics::kSubsetBlock / (members * n * n));
    Shape shape = G.shape();
    shape[batch] = shape[batch + 1] = 1;
    Tensor sum = G.zeros_like(shape);
    Combinations subsets(m, n);
    for (std::vector<int64_t> flat = subsets.next(block); !flat.empty();
         flat = subsets.next(block)) {
        const int64_t count = int64_t(flat.size()) / n;
        std::vector<double> indices(flat.size());
        std::transform(flat.begin(), flat.end(), indices.begin(),
                       [](int64_t i) { return static_cast<double>(i); });
        const Tensor chosen = Tensor::like(G, indices, {count, n});
        sum = sum + priv_abs_det(priv_column_subsets(G, chosen)).sum(int(batch));
    }
    return sum;
}

/// sqrt(det(G G')) for G (..., n, m) of rank n, as (..., 1, 1), by the Cholesky factor of G G'; the
/// Gram matrices of the members `skip` marks are replaced by the identity.
Tensor aux_gram_volume(const Tensor &G, const Tensor &skip) {
    const Tensor gram = G.matmul(G.matrix_transpose());
    const Tensor safe = Tensor::where(skip, gram.eye_like(gram.shape().back()), gram);
    return safe.cholesky().diagonal().prod(-2, true);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::volume(const std::string &method, int order) const {
    detail::Entry entry;
    CORA_CHECK(
        method == "exact" || method == "reduce" || method == "alamo", Error::wrong_value, [&] {
            return "volume: the method must be 'exact', 'reduce' or 'alamo', got '" + method + "'";
        });
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "volume: the order must be a positive integer, got " + std::to_string(order);
    });
    CORA_CHECK(method != "reduce", Error::not_supported, [] {
        return std::string("volume: the method 'reduce' needs the order reduction 'pca', which "
                           "reduce does not have");
    });
    const int64_t n = dim(), m = G_.shape().back();
    Shape shape = c_.shape();
    shape[shape.size() - 2] = 1;
    const Tensor zero = c_.zeros_like(shape);
    if (m < n) return zero;
    const Tensor deficient = aux_rank_deficient(G_);
    // the exact or the Gram volume, zero if the generators are deficient
    const Tensor value =
        method == "exact" ? aux_sum_of_determinants(G_) : aux_gram_volume(G_, deficient);
    // 2^n is a power of two: the scaling is exact
    return Tensor::where(deficient, zero, value * std::ldexp(1.0, int(n)));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
