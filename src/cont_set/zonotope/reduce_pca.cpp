// reduce_pca - CORA's zonotope.reduce with method "pca" and order 1
//
// A zonotope with more nonzero generators than dimensions becomes the box of its generators in
// their principal axes: the axes are the left singular vectors of G G' (the covariance of the
// points [G, -G] up to its scale), the generators the box of G in them, rotated back. With at
// most n nonzero generators the zonotope stays. The center stays.
//
// Syntax:   Zred = Z.reduce_pca();
// Inputs:   -
// Outputs:  Zred - a zonotope with at most n generators that encloses Z
// See also: reduce, reduce_adaptive
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reducePCA.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce_pca() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2 && G_.ndim() == 2, Error::not_supported,
               [] { return std::string("reduce_pca: not for a batch of zonotopes"); });
    // the nonzero generators; with at most n of them nothing is reduced
    const std::vector<double> size = G_.abs().sum(0).data();
    std::vector<int64_t> keep;
    for (std::size_t j = 0; j < size.size(); ++j)
        if (size[j] > 0) keep.push_back(int64_t(j));
    if (int64_t(keep.size()) <= dim()) return *this;
    const Tensor G = G_.take(keep, -1);
    // the axes: the covariance of the points [G, -G] has them; its scale does not matter
    const Tensor U = G.matmul(G.matrix_transpose()).svd()[0];
    const Tensor box = U.matrix_transpose().matmul(G).abs().sum(-1, true);
    return {c_, U.matmul(box.diag())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
