// priv_null_space - an orthonormal basis of the null space of a matrix, as MATLAB's null
//
// Syntax:   Tensor N = priv_null_space(A);
// Inputs:   A - a matrix (p, m)
// Outputs:  N - (m, k): the right singular vectors of the singular values that are at most
//           max(p, m) eps times the largest; k = 0 if A has full column rank
// See also: priv_linear_programs, is_full_dim
//
// CORA.cpp only: MATLAB calls null(A)

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_null_space(const Tensor &A) {
    const int64_t p = A.shape()[0], m = A.shape()[1];
    // a wide matrix gets zero rows, so that the factorization has all m right singular vectors
    const Tensor square = p < m ? Tensor::concat({A, priv_zeros(A, m - p, m)}, 0) : A;
    const std::array<Tensor, 3> usv = square.svd();
    const std::vector<double> s = usv[1].data();
    const double largest = s.empty() ? 0.0 : s.front();
    const double tol = double(std::max(p, m)) * machine_eps(A.dtype()) * largest;
    const int64_t rank = std::count_if(s.begin(), s.end(), [&](double x) { return x > tol; });
    return usv[2].slice(-1, rank, m);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
