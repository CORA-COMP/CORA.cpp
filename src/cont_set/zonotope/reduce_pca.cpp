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
// References:
//    [1] A. Girard. "Reachability of uncertain linear systems using zonotopes", HSCC 2005, LNCS
//        3414, pp. 291-305.
//    [2] A.-K. Kopetzki, B. Schurmann, M. Althoff. "Methods for order reduction of zonotopes", CDC
//        2017.
//    [3] C. Combastel. "A state bounding observer based on zonotopes", ECC 2003.
//    [4] J. K. Scott, D. M. Raimondo, G. R. Marseglia, R. D. Braatz. "Constrained zonotopes: a new
//        tool for set-based estimation and fault detection", Automatica 69, 2016, pp. 126-136.
//    [5] M. Wetzlinger, A. Kulmburg, M. Althoff. "Adaptive parameter tuning for reachability
//        analysis of nonlinear systems", HSCC 2021.
//    [6] S. Sadraddini, R. Tedrake. "Linear encodings for polytope containment problems", CDC
//        2019.
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reducePCA.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce_pca() const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 1 && G_.ndim() == 2, Error::not_supported,
               [] { return std::string("reduce_pca: not for a batch of zonotopes"); });
    // the nonzero generators; with at most n of them nothing is reduced
    const std::vector<double> size = G_.abs().sum(-1).data();
    std::vector<int64_t> keep;
    for (std::size_t j = 0; j < size.size(); ++j)
        if (size[j] > 0) keep.push_back(int64_t(j));
    if (int64_t(keep.size()) <= dim()) return *this;
    const Tensor G = G_.take(keep, -2); // the rows of the kept generators
    // the axes: the covariance of the points [G, -G] has them; its scale does not matter
    const Tensor U = G.matrix_transpose().matmul(G).svd()[0];
    const Tensor box = G.matmul(U).abs().sum(-2).expand_dims(-1);
    // the generators U diag(box) as rows diag(box) U'
    return {c_, box.diag().matmul(U.matrix_transpose())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
