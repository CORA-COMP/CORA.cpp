// expm_absolute_bound - CORA's absolute bound of the mapping mimicking exp (expmAbsoluteBound)
//
// A is square, with the spectral radii of |C| t / 2 and (|C| + S) t / 2 below 1 (C the center, S
// the radius), which the checks verify: then the eigenvalue 1 of CORA's block matrix is the n-fold
// one of its last block. G is the block matrix [|C| t / 2, 0, 0; S t / 2, (|C| + S) t / 2, 0; 0, I,
// I]. CORA forms the spectral projector for the eigenvalue 1 from the eigenvectors of G and G';
// here the same limit comes from the block structure by two solves (no eigendecomposition).
// The check B x < x, x = (I - B)^-1 1 > 0 (Collatz-Wielandt) is sound at nearest, u the unit
// roundoff: B_up = fl(B (1 + 16 u)) is above the true (|C| + S) t / 2 (5 roundings of at most
// 5.1 u), and fl(B_up x) (1 + 4 n u) >= B_up x because a sum of n nonnegative products loses at
// most gamma_n; a NaN fails every comparison.
//
// Syntax:   Tensor B = mat_exp::expm_absolute_bound(A, t);
// Inputs:   t >= 0 - the time
// Outputs:  B - (..., n, n): the last block row of lim G^k times [|C| t; S t; 0]
// See also: expm_norm_infinity, abs
//
// CORA: matrixSet/@intervalMatrix/expmAbsoluteBound.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// True if the spectral radius of the nonnegative B (n x n, every member of a batch) is below 1.
bool aux_spectral_radius_below_one(const Tensor &B, const Tensor &I) {
    const double u = machine_eps() / 2;
    const int64_t n = B.shape()[B.ndim() - 1];
    const Tensor B_up = B * (1.0 + 16 * u);
    const Tensor x = (I - B_up).solve(B.ones_like().max(-1, true));
    const Tensor y = B_up.matmul(x) * (1.0 + 4 * double(n) * u);
    return x.min().item() > 0 && y.less(x).min().item() > 0;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor mat_exp::expm_absolute_bound(const Interval &A, double t) {
    detail::Entry entry;
    CORA_CHECK(A.inf().size(-2) == A.inf().size(-1), Error::dimension_mismatch, [&] {
        return "expm_absolute_bound: the matrix must be square, got the shape " +
               shape_text(A.inf().shape());
    });
    CORA_CHECK(t >= 0, Error::wrong_value, [&] {
        return "expm_absolute_bound: the time must be nonnegative, got " + std::to_string(t);
    });
    const Tensor C = A.center(), S = A.rad(), I = C.eye_like(A.inf().size(-2));
    const Tensor abs_C = C.abs();
    const double j = 2.0;
    const Tensor B1 = abs_C * t / j, B2 = (abs_C + S) * t / j, C21 = S * t / j;
    CORA_CHECK(aux_spectral_radius_below_one(B2, I), Error::wrong_value, [&] {
        return "expm_absolute_bound: the spectral radius of (|C| + S) t / 2 must be below 1";
    });
    // x -> B1 x, y -> C21 x + B2 y, z -> y + z: the sums of the x and y sequences
    const Tensor Y = (I - B1).solve(abs_C * t);
    return (I - B2).solve(C21.matmul(Y) + S * t);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
