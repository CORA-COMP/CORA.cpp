// is_bad_dir - which directions are bad for the Minkowski difference of two ellipsoids
//
// lambda is the largest generalized eigenvalue of Q2 against Q1, tol the least tolerance of the
// two.
//
// Syntax:   std::vector<bool> res = E1.is_bad_dir(E2, L);
// Inputs:   E2 - an ellipsoid of the dimension of E1, E1 with Q positive definite;
//           L - directions as columns (n, k). Not a batch
// Outputs:  res - per column l: sqrt(l' Q1 l) / sqrt(l' Q2 l) > 1 / lambda + tol
// See also: is_bigger
//
// CORA: contSet/@ellipsoid/isBadDir.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<bool> Ellipsoid::is_bad_dir(const Ellipsoid &E2, const Tensor &L) const {
    detail::Entry entry;
    detail::Refusal refusal = priv_pair_refusal(*this, E2, "is_bad_dir");
    if (!refusal && !(L.defined() && L.ndim() == 2 && L.shape()[0] == dim()))
        refusal = std::pair{Error::wrong_value, "is_bad_dir: L must have " + std::to_string(dim()) +
                                                    " rows, one column per direction"};
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const double tol = std::min(tol_, E2.tol());
    // 1 / the largest generalized eigenvalue
    const double r = 1.0 / priv_simdiag(Q_, E2.Q(), tol).second.diagonal().max().item();
    const Tensor q1 = L.multiply(Q_.matmul(L)).sum(0, true);
    const Tensor q2 = L.multiply(E2.Q().matmul(L)).sum(0, true);
    const std::vector<double> ratio = q1.sqrt().divide(q2.sqrt()).data();
    std::vector<bool> bad;
    bad.reserve(ratio.size());
    for (const double x : ratio) bad.push_back(x > r + tol);
    return bad;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
