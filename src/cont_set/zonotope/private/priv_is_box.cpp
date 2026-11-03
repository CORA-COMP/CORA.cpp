// priv_is_box - whether a zonotope is a box
//
// Syntax:   bool box = priv_is_box(Z, tol);
// Inputs:   Z - zonotope;  tol - generator entries up to tol count as zero (as within_tol)
// Outputs:  box - true if every generator has at most one entry beyond tol, or Z is
//           one-dimensional; of a batch, every member
// See also: Zonotope::intersect, Zonotope::contains
//
// CORA: contSet/@zonotope/representsa_.m

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool priv_is_box(const Zonotope &Z, double tol) {
    if (Z.dim() == 1) return true;
    const Tensor G = Z.generators().detach();
    const Tensor beyond = within_tol(G, G.zeros_like(), tol).negative() + 1.0;
    const Tensor per_generator = beyond.sum(-2);
    return per_generator.numel() == 0 || per_generator.max().item() <= 1;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
