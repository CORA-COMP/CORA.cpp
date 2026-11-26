// compact - the constrained polynomial zonotope without redundancy, as CORA's compact_
//
// In CORALean's order: the generators of equal exponents merged (PolyZonotope::compact_gens), the
// monomials of equal exponents merged (PolyLevelSet::compact), then the factors that neither the
// generators nor the constraints read dropped. No zero generator goes and no constant moves into
// the center (CORA's "all" does both).
//
// Syntax:   S = cPZ.compact();
// Inputs:   method - "all" (default)
//           tol - checked, unused
// Outputs:  S - the same set, with fewer columns
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/compact_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::compact(const std::optional<std::string> &method, Tol tol) const {
    detail::Entry entry;
    (void)resolve_tol(entry, tol, c().dtype(), "compact");
    CORA_CHECK(!method || *method == "all", Error::no_specific_alg,
               [&] { return "compact: the method must be \"all\", got \"" + *method + "\""; });
    const PolyZonotope P = poly_.compact_gens();
    const PolyLevelSet C = cons_.compact();
    // the factors some generator or monomial reads, in their order
    std::vector<int64_t> kept, id;
    for (std::size_t k = 0; k < ids().size(); ++k) {
        bool used = false;
        for (const std::vector<int> &col : P.exponents()) used = used || col[k] != 0;
        for (const std::vector<int> &col : C.exponents()) used = used || col[k] != 0;
        if (used) {
            kept.push_back(int64_t(k));
            id.push_back(ids()[k]);
        }
    }
    // the exponent columns on the kept factors
    Exponents E;
    for (const std::vector<int> &col : P.exponents()) {
        std::vector<int> e;
        e.reserve(kept.size());
        for (const int64_t k : kept) e.push_back(col[std::size_t(k)]);
        E.push_back(std::move(e));
    }
    return {PolyZonotope(P.center(), P.generators(), P.independent_generators(), std::move(E),
                         std::move(id), detail::Verbatim{}),
            C.reindex_vars(kept)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
