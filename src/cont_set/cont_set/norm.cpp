// norm - an upper bound on the norm of the points of a set, as CORA's contSet.norm
//
// Which types and modes a class offers is its own (a box ignores the method). Error::noops for a
// class without it.
//
// Syntax:   val = S.norm();   val = S.norm(type, method);
// Inputs:   S - set
//           type - one, two (default), inf or fro; method - "ub" (default), "exact" or "ub_convex"
// Outputs:  val - the bound, (...); -Inf for the empty set
// See also: Interval::norm, Zonotope::norm
//
// CORA: contSet/@contSet/norm.m
// CORA: contSet/@contSet/norm_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::norm(Tensor::Norm type, const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(
        method == "exact" || method == "ub" || method == "ub_convex", Error::wrong_value, [&] {
            return "norm: the method must be 'exact', 'ub' or 'ub_convex', got '" + method + "'";
        });
    entry.release();
    return norm_impl(type, method);
}

Tensor ContSet::norm_impl(Tensor::Norm, const std::string &) const { priv_noops("norm", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
