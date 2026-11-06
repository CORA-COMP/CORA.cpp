// norm - an upper bound on the norm of the points of a set, as CORA's contSet.norm
//
// Syntax:   val = S.norm();   val = S.norm(type, mode);
// Inputs:   S - set
//           type - one, two (default), inf or fro;  mode - "ub" (default), "exact" or
//           "ub_convex"; which of them a class offers is its own (a box ignores the mode)
// Outputs:  val - the bound, (..., 1, 1); -Inf for the empty set; Error::noops for a class
//           without it
// See also: Interval::norm, Zonotope::norm
//
// CORA: contSet/@contSet/norm.m
// CORA: contSet/@contSet/norm_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::norm(Tensor::Norm type, const std::string &mode) const {
    detail::Entry entry;
    CORA_CHECK(mode == "exact" || mode == "ub" || mode == "ub_convex", Error::wrong_value, [&] {
        return "norm: the mode must be 'exact', 'ub' or 'ub_convex', got '" + mode + "'";
    });
    return norm_impl(type, mode);
}

Tensor ContSet::norm_impl(Tensor::Norm, const std::string &) const { priv_noops("norm", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
