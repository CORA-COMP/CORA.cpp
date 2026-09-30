// priv_inverse - the inverse of a square matrix, empty if it is singular (MATLAB's rank test)
//
// Gauss-Jordan elimination with partial pivoting on the host; a pivot below size * eps * max|A|
// counts as zero, which is the tolerance scale of MATLAB's rank.
//
// Syntax:   std::optional<Tensor> Ainv = priv_inverse(A);
// Inputs:   A - matrix (n, n)
// Outputs:  Ainv - the inverse, or empty if A is (numerically) singular
// See also: priv_verifyRA_supportFunc, priv_reach_adaptive

#include "contDynamics/linearSys/private/priv_verify.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::optional<Tensor> priv_inverse(const Tensor &A) {
    const int64_t n = A.shape()[0];
    std::vector<double> M = A.data(), inv(n * n, 0.0);
    double maxAbs = 0;
    for (double v : M) maxAbs = std::max(maxAbs, std::abs(v));
    for (int64_t i = 0; i < n; ++i) inv[i * n + i] = 1;
    const double tol = double(n) * 2.220446049250313e-16 * maxAbs;
    for (int64_t j = 0; j < n; ++j) {
        // The largest entry of the column below the diagonal is the pivot.
        int64_t piv = j;
        for (int64_t i = j + 1; i < n; ++i)
            if (std::abs(M[i * n + j]) > std::abs(M[piv * n + j])) piv = i;
        if (std::abs(M[piv * n + j]) <= tol) return std::nullopt;
        if (piv != j) {
            std::swap_ranges(M.begin() + j * n, M.begin() + (j + 1) * n, M.begin() + piv * n);
            std::swap_ranges(inv.begin() + j * n, inv.begin() + (j + 1) * n, inv.begin() + piv * n);
        }
        const double d = M[j * n + j];
        for (int64_t c = 0; c < n; ++c) {
            M[j * n + c] /= d;
            inv[j * n + c] /= d;
        }
        for (int64_t i = 0; i < n; ++i) {
            const double f = M[i * n + j];
            if (i == j || f == 0) continue;
            for (int64_t c = j; c < n; ++c) M[i * n + c] -= f * M[j * n + c];
            for (int64_t c = 0; c < n; ++c) inv[i * n + c] -= f * inv[j * n + c];
        }
    }
    return Tensor::like(A, inv, {n, n});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
