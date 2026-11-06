// representsa - whether a zonotope is also a set of another type, as CORA's zonotope.representsa_
//
// The norm of a zero generator is up to the machine epsilon. The converted set (CORA's second
// output) is not returned: use the conversions. "conHyperplane" and "ellipsoid" are refused
// (Error::not_supported), as CORA does, and so is "polygon", for which CORA's switch has no case.
// Not a batch.
//
// Syntax:   bool res = Z.representsa(type);   bool res = Z.representsa(type, tol);
// Inputs:   type - CORA's name of a set type ("origin", "point", "interval", "zonotope", ...)
//           tol - what counts as zero; default: the repository tolerance
// Outputs:  res - whether the set is exactly such a set, judged without its zero generators
// See also: is_full_dim, rank, compact
//
// CORA: contSet/@zonotope/representsa_.m
// CORA: contSet/@contSet/representsa.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/within_tol.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The types of CORA's representsa.
const char *const kTypes[] = {
    "capsule",    "conPolyZono", "conHyperplane", "conZonotope", "ellipsoid",    "halfspace",
    "interval",   "levelSet",    "polygon",       "polytope",    "polyZonotope", "probZonotope",
    "zonoBundle", "zonotope",    "origin",        "point",       "hyperplane",   "parallelotope",
    "convexSet",  "emptySet",    "fullspace"};

/// Whether the 0/1 mask holds everywhere.
bool aux_all(const Tensor &mask) { return mask.numel() == 0 || mask.min().item() != 0; }

} // namespace


// ===========================================  MAIN  =========================================== //

bool Zonotope::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(std::find(std::begin(kTypes), std::end(kTypes), type) != std::end(kTypes),
               Error::wrong_value,
               [&] { return "representsa: '" + type + "' is not a type of set that CORA knows"; });
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "representsa");
    CORA_CHECK(c_.ndim() == 2, Error::not_supported, [] {
        return std::string("representsa: it supports a single zonotope, not a batch");
    });
    CORA_CHECK(type != "conHyperplane" && type != "ellipsoid" && type != "polygon",
               Error::not_supported, [&] {
                   return "representsa: the comparison of a zonotope with '" + type +
                          "' is not supported";
               });
    // CORA judges the zonotope without its zero generators
    const Zonotope Z = compact("zeros", machine_eps(c_.dtype()));
    const int64_t n = Z.dim(), gens = Z.G_.shape().back();
    const auto norm_of_radius = [&] {
        return Z.G_.abs_sum().vector_norm(Tensor::Norm::two, -2, true).item();
    };
    // the special types first, then the sets that every zonotope is and the ones that none is
    if (type == "origin")
        return (gens == 0 && aux_all(within_tol(Z.c_, Z.c_.zeros_like(), tol))) ||
               Z.interval().norm().item() <= tol;
    if (type == "point") return gens == 0 || norm_of_radius() <= tol;
    if (type == "capsule") return n == 1 || gens <= 1;
    if (type == "interval") return priv_is_box(Z, tol);
    if (type == "parallelotope") return n == 1 || (gens == n && Z.is_full_dim());
    // the sets that every zonotope is, and the ones that none is
    return type != "halfspace" && type != "probZonotope" && type != "hyperplane" &&
           type != "emptySet" && type != "fullspace";
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
