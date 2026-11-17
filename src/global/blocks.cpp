// blocks - an orthonormal basis of the null space of a matrix, as MATLAB's null
//
// The threshold is relative to the largest singular value; k = 0 if A has full column rank.
//
// Syntax:   Tensor N = null_space(A);
// Inputs:   A - a matrix (p, m)
// Outputs:  N - (m, k): the right singular vectors of the singular values at most max(p, m) eps
// See also: blocks.h
//
// CORA.cpp only: MATLAB calls null(A)
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "global/blocks.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <array>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor null_space(const Tensor &A) {
    const int64_t p = A.shape()[0], m = A.shape()[1];
    // a wide matrix gets zero rows, so that the factorization has all m right singular vectors
    const Tensor square = p < m ? Tensor::concat({A, zeros_block(A, m - p, m)}, 0) : A;
    const std::array<Tensor, 3> usv = square.svd();
    const std::vector<double> s = usv[1].data();
    const double largest = s.empty() ? 0.0 : s.front();
    const double tol = double(std::max(p, m)) * machine_eps(A.dtype()) * largest;
    const int64_t rank = std::count_if(s.begin(), s.end(), [&](double x) { return x > tol; });
    return usv[2].slice(-1, rank, m);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
