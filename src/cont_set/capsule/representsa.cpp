// representsa - whether a capsule is a set of another class, as CORA's capsule.representsa_
//
// Decides the types CORA answers without a conversion: a point or the origin up to tol, an
// interval, a zonotope or a polytope when the radius or the generator vanishes, and so on. The
// empty capsule follows representsa_emptyObject. Types that CORA refuses (conPolyZono, levelSet,
// ...) are refused.
//
// Syntax:   res = C.representsa(type);   res = C.representsa(type, tol);
// Inputs:   type - CORA's name of the class ("origin", "point", "zonotope", ...)
//           tol - relaxes the zero tests, default: the repository tolerance (docs/architecture.md)
// Outputs:  res - whether the capsule is such a set
// See also: isequal, is_full_dim
//
// CORA: contSet/@capsule/representsa_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cont_set/cont_set/private/priv_set_type.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Capsule::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(priv_is_set_type(type), Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "representsa");
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("representsa: it supports a single capsule, not a batch");
    });
    CORA_CHECK(!priv_type_in(type, {"conPolyZono", "levelSet", "polyZonotope", "parallelotope"}),
               Error::not_supported, [&] {
                   return "representsa: the comparison of a capsule with '" + type +
                          "' is not supported";
               });
    const int64_t n = dim();
    // the empty capsule is what representsa_emptyObject says
    if (is_empty()) return !priv_type_in(type, {"origin", "point", "fullspace"});
    const auto zero = [&](const Tensor &x) { return priv_all(within_tol(x, x.zeros_like(), tol)); };
    const bool segment_is_point = zero(g_), no_radius = zero(r_);
    const bool is_point = segment_is_point && no_radius;
    if (type == "origin") return is_point && zero(c_);
    if (type == "point") return is_point;
    if (type == "interval") {
        // a box has all but one of the generator's entries at zero
        const Tensor flat = within_tol(g_, g_.zeros_like(), tol);
        return n == 1 || (no_radius && flat.sum().item() >= static_cast<double>(n - 1));
    }
    if (type == "ellipsoid") return n == 1 || segment_is_point || no_radius;
    if (priv_type_in(type, {"conHyperplane", "conZonotope", "polytope", "zonoBundle", "zonotope"}))
        return n == 1 || no_radius;
    if (type == "hyperplane") return n == 1;
    // the sets that every capsule is, and the ones that none is
    return type == "capsule" || type == "convexSet";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
