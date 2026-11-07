// complement - the complement of a polytope, as CORA's polytope.not
//
// Defined for one inequality (and no equality), the empty set and R^n. The complement of
// {a x <= b} is not closed; CORA takes the closed halfspace {-a x <= -b - eps}, which lies inside
// it. The complement of the empty set is R^n and of R^n the empty set. Anything else is
// Error::not_supported. Not a batch.
//
// Syntax:   Pres = P.complement();
// Inputs:   -
// Outputs:  Pres - the complement, up to eps for a halfspace
// See also: Polytope::empty, Polytope::inf, EmptySet::complement
//
// CORA: contSet/@polytope/not.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::complement() const {
    detail::Entry entry;
    const Constraints C = constraints();
    CORA_CHECK(C.A.ndim() == 2, Error::not_supported, [] {
        return std::string("complement: it supports a single polytope, not a batch");
    });
    // one inequality: the opposite halfspace, moved off the boundary by eps
    if (C.be.numel() == 0 && C.b.numel() == 1) return {-C.A, -C.b - machine_eps(C.b.dtype())};
    const double eps = machine_eps(C.b.dtype());
    if (representsa("emptySet", eps)) return inf(dim());
    if (representsa("fullspace", eps)) return empty(dim());
    CORA_FAIL(Error::not_supported, "complement: it is supported for a single inequality, the "
                                    "empty set and the whole space only");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
