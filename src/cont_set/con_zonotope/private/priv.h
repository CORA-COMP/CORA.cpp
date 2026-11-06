// priv - the private functions of con_zonotope, as CORA's @conZonotope/private
//
// Not for callers of the class: the operations of the class and the sets that convert to it use
// them. The linear programs of the class run on the host, on one (unbatched) set.
//
// Syntax:   Tensor Z = priv_zeros(c, 0, 3);   HostConZonotope H = priv_host_con_zonotope(cz);
//           bool empty = priv_is_empty(H);
// See also: con_zonotope.h
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"

#include <optional>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Blocks ------------------------------------------------------------------------------------------

/// Zeros (..., rows, cols) with the batch shape and the data of `like`.
Tensor priv_zeros(const Tensor &like, int64_t rows, int64_t cols);

/// The constant matrix (..., rows, cols) of one value, with the batch shape and the data of `like`.
Tensor priv_full(const Tensor &like, int64_t rows, int64_t cols, double value);

/// The identity (..., k, k) with the batch shape of `like`.
Tensor priv_identity(const Tensor &like, int64_t k);

/// The block matrix [[A, 0], [0, B]] (CORA's blkdiag) of two matrices of one batch shape.
Tensor priv_block_diag(const Tensor &A, const Tensor &B);

/// The columns of A (..., p, m) at the positions of `columns` kept, the others dropped.
Tensor priv_select_columns(const Tensor &A, const std::vector<int64_t> &columns);

/// Why a binary operation of two constrained zonotopes is refused, if it is: the sets must have
/// one batch shape and, if same_dim, one dimension. `operation` names it in the message.
detail::Refusal priv_pair_refusal(const char *operation, const ConZonotope &a, const ConZonotope &b,
                                  bool same_dim);

/// The rows of [A, b] that are not all zero, as host indices (of a batch, those that some member
/// keeps).
std::vector<int64_t> priv_nonzero_rows(const Tensor &A, const Tensor &b);

/// The rows and generator columns of cz that CORA's compact_ 'zeros' keeps: the rows of [A, b]
/// that are not all zero and the columns of [G; A] that are not all zero (of a batch, those that
/// some member keeps), as host indices.
std::pair<std::vector<int64_t>, std::vector<int64_t>> priv_nonzero_parts(const ConZonotope &cz);

/// cz without the constraint rows 0 = 0 and the generators that are zero and in no constraint
/// (CORA's compact_ 'zeros'); only exact zeros count.
ConZonotope priv_compact(const ConZonotope &cz);

/// The set M cz + t (CORA's mtimes with a matrix and plus with a point): the linear image of the
/// center and the generators, the constraints unchanged; t may be empty (no shift).
ConZonotope priv_affine_image(const ConZonotope &cz, const Tensor &M,
                              const std::optional<Tensor> &t);

/// The Minkowski sum of two constrained zonotopes: centers add, generators join, constraints on
/// the block diagonal.
ConZonotope priv_minkowski_sum(const ConZonotope &cz1, const ConZonotope &cz2);

/// The points x of cz1 whose image (Mc + MG x) is a point of cz2, as CORA's conIntersect: the
/// factors of both stack and one equality per dimension makes them agree; Mc (n2, 1) and MG
/// (n2, m1) are the center and generators of the image of cz1.
ConZonotope priv_agreement(const ConZonotope &cz1, const ConZonotope &cz2, const Tensor &Mc,
                           const Tensor &MG);

// Corners and vertices ----------------------------------------------------------------------------

/// Points of a single constrained zonotope that contain its vertices, as columns (n, K): none if
/// the set is empty. A convex set contains the set exactly if it contains these points.
Tensor priv_vertex_candidates(const ConZonotope &cz);

// Linear programs ---------------------------------------------------------------------------------

/// A single constrained zonotope on the host: c (n), G (n x m), A (p x m) and b (p), row-major.
struct HostConZonotope {
    int64_t n = 0, m = 0, p = 0;
    std::vector<double> c, G, A, b;
};

/// The values of a single (unbatched) constrained zonotope on the host.
HostConZonotope priv_host_con_zonotope(const ConZonotope &cz);

/// The least and the largest value of d'(c + G x) over the factors x with |x|_inf <= 1 and
/// A x = b, a program each; none if there is no such x (CORA's supportFunc_ with 'range').
std::optional<std::pair<double, double>> priv_range(const HostConZonotope &H,
                                                    const std::vector<double> &d);

/// Whether no x has |x|_inf <= 1 and A x = b (CORA's representsa_(cz, 'emptySet')).
bool priv_is_empty(const HostConZonotope &H);

/// The least and the largest value of every factor over the nonempty polytope (2 m programs).
std::pair<std::vector<double>, std::vector<double>> priv_factor_ranges(const HostConZonotope &H);

/// The center of the largest ball in {x | Aineq x <= bineq, Aeq x = beq} (CORA's polytope center,
/// 'chebyshev'): the matrices are row-major with k columns; none if the set is empty.
std::optional<std::vector<double>> priv_chebyshev_center(const std::vector<double> &Aineq,
                                                         const std::vector<double> &bineq,
                                                         const std::vector<double> &Aeq,
                                                         const std::vector<double> &beq, int64_t k);

/// An orthonormal basis (m, k) of the null space of A (p, m) (MATLAB's null): the right singular
/// vectors of the singular values up to max(p, m) eps times the largest.
Tensor priv_null_space(const Tensor &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
