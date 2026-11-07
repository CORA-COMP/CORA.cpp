// enclose - an ellipsoid that contains an ellipsoid and the image of another, as CORA's enclose
//
// E2 = M Eplus; the ellipsoids are single and nonempty. CORA finds the union by an SDP solver;
// none is available, so the call is Error::not_supported after the checks. CORA's help says E2 =
// M E + Eplus, its code takes M Eplus (docs/cora-differences.md); the code's form is kept.
//
// Syntax:   Ellipsoid R = E.enclose(E2);   Ellipsoid R = E.enclose(M, Eplus);
// Inputs:   E2 - an ellipsoid of the dimension of E, or M (n, m) and Eplus of dimension m
// Outputs:  R - would be the outer union: the minimum-volume ellipsoid that contains E and E2
// See also: Zonotope::enclose, intersect
//
// CORA: contSet/@ellipsoid/enclose.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   07-November-2026
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The failure of every form: no SDP solver for the minimum-volume union (no return, so that no
/// caller has code after it).
[[noreturn]] void aux_no_sdp_solver() {
    CORA_FAIL(Error::not_supported,
              "enclose: the outer union of two ellipsoids is a minimum-volume problem for an SDP "
              "solver, which is not available");
}

} // namespace


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::enclose(const Ellipsoid &E2) const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_pair_refusal(*this, E2, "enclose");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    aux_no_sdp_solver();
}

Ellipsoid Ellipsoid::enclose(const Tensor &M, const Ellipsoid &Eplus) const {
    detail::Entry entry;
    CORA_CHECK(M.defined() && M.ndim() == 2 && M.shape()[0] == dim() && M.shape()[1] == Eplus.dim(),
               Error::wrong_value, [&] {
                   return "enclose: M must be a matrix (" + std::to_string(dim()) + ", " +
                          std::to_string(Eplus.dim()) + ")";
               });
    detail::Refusal refusal = priv_unbatched_refusal(Eplus, "enclose");
    if (!refusal && priv_is_empty(Eplus))
        refusal = std::pair{Error::empty_set, std::string("enclose: Eplus is empty")};
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // the second ellipsoid is M Eplus
    const Ellipsoid E2(priv_congruence(M, Eplus.Q()), M.matmul(Eplus.q()), Eplus.tol());
    refusal = priv_pair_refusal(*this, E2, "enclose");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    aux_no_sdp_solver();
}

Set Ellipsoid::enclose_impl(const ContSet &S2) const {
    if (S2.class_id() != ClassId::ellipsoid) priv_noops("enclose", *this, &S2);
    aux_no_sdp_solver();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
