// reduce - fewer generators for a zonotope, as CORA's zonotope.reduce with method "girard"
//
// The generators that cost the most to box stay, by Girard's metric ||g||_1 - ||g||_inf (zero on an
// axis-aligned one); the others are replaced by the box that encloses them. The result contains the
// zonotope and has at most order * n generators (n the dimension). A batch reduces member by
// member.
//
// Syntax:   Zred = Z.reduce(order);
// Inputs:   order - the largest number of generators per dimension, at least 1
// Outputs:  Zred - a zonotope that encloses Z; Z itself if it has few enough generators
// See also: interval, plus
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
// CORA: contSet/@zonotope/private/priv_reduceGirard.m
// CORA: matrixSet/@matZonotope/reduce.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce(int order) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the zonotope order must be at least 1, got " + std::to_string(order);
    });
    const int64_t n = dim(), m = nr_generators();
    if (m <= order * n) return *this;

    // the longest generators stay and the rest becomes a box
    if (elem_rank_ == 1) return {c_, priv_girard_generators(G_, order * n - n)};
    // a matrix zonotope as its stacked matrices (CORA's matZonotope.reduce), back column by column
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("reduce: it supports a single set, not a batch"); });
    const Zonotope Z = stacked().reduce(order);
    const int64_t rows = c_.size(-2), cols = c_.size(-1), h = Z.nr_generators();
    return matrix(Z.center().reshape({cols, rows}).matrix_transpose(),
                  Z.generators().reshape({h, cols, rows}).matrix_transpose());
}

Set Zonotope::reduce_impl(int order) const { return reduce(order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
