// enclose - a zonotope bundle that encloses a bundle and a linear image of it, as CORA's enclose
//
// Member by member (Zonotope::enclose). With a map, member i of the second set is M Z_i + S for a
// zonotope S, and M Z_i + the zonotope of the box around S for a bundle S (CORA's plus).
//
// Syntax:   ZonoBundle R = bundle.enclose(S);   ZonoBundle R = bundle.enclose(M, S);
// Inputs:   S - a bundle with as many members (any number with a map) or a zonotope
//           M - a matrix (..., n, n): the second set is M bundle + S
// Outputs:  R - a bundle that encloses the convex hull of the union
// See also: Zonotope::enclose, intersect
//
// CORA: contSet/@zonoBundle/enclose.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Checks ------------------------------------------------------------------------------------------

/// Why M is refused, if it is: it is not (..., n, n) of the dimension of the bundle.
detail::Refusal aux_matrix_refusal(const ZonoBundle &bundle, const Tensor &M) {
    if (M.ndim() >= 2 && M.size(-2) == bundle.dim() && M.size(-1) == bundle.dim())
        return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "enclose: M must be square of the dimension " + std::to_string(bundle.dim()) +
                         " of the bundle, got the shape " + shape_text(M.shape())};
}

/// Why a bundle S with as many members is required, if it is not given.
detail::Refusal aux_count_refusal(const ZonoBundle &bundle, const ZonoBundle &S) {
    if (S.parallel_sets() == bundle.parallel_sets()) return std::nullopt;
    return std::pair{Error::wrong_value,
                     "enclose: the bundles must have the same number of zonotopes, got " +
                         std::to_string(bundle.parallel_sets()) + " and " +
                         std::to_string(S.parallel_sets())};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Bundles and zonotopes ---------------------------------------------------------------------------

ZonoBundle ZonoBundle::enclose(const ZonoBundle &S) const {
    detail::Entry entry;
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "enclose");
    if (!r) r = aux_count_refusal(*this, S);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // member i of the result encloses member i of both bundles
    std::vector<Zonotope> hulls;
    hulls.reserve(Z_.size());
    for (std::size_t i = 0; i < Z_.size(); ++i) hulls.push_back(Z_[i].enclose(S.zonotopes()[i]));
    return ZonoBundle(std::move(hulls));
}

ZonoBundle ZonoBundle::enclose(const Zonotope &Z) const {
    detail::Entry entry;
    const detail::Refusal r = priv_operand_refusal(*this, Z.dim(), Z.center(), "enclose");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the zonotope is enclosed with every member
    std::vector<Zonotope> hulls;
    hulls.reserve(Z_.size());
    for (const Zonotope &A : Z_) hulls.push_back(A.enclose(Z));
    return ZonoBundle(std::move(hulls));
}

// With a map and a shift --------------------------------------------------------------------------

ZonoBundle ZonoBundle::enclose(const Tensor &M, const Zonotope &Zplus) const {
    detail::Entry entry;
    detail::Refusal r = priv_operand_refusal(*this, Zplus.dim(), Zplus.center(), "enclose");
    if (!r) r = aux_matrix_refusal(*this, M);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // member i of the second set is M Z_i + Zplus
    std::vector<Zonotope> hulls;
    hulls.reserve(Z_.size());
    for (const Zonotope &A : Z_) hulls.push_back(A.enclose(M, Zplus));
    return ZonoBundle(std::move(hulls));
}

ZonoBundle ZonoBundle::enclose(const Tensor &M, const ZonoBundle &S) const {
    detail::Entry entry;
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "enclose");
    if (!r) r = aux_matrix_refusal(*this, M);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // CORA's plus replaces the bundle by the zonotope of its box
    const Interval box = S.interval();
    CORA_CHECK(!box.is_empty(), Error::not_supported,
               [] { return std::string("enclose: the second bundle is empty"); });
    const Zonotope around = box.zonotope();
    // then it is a zonotope shift
    std::vector<Zonotope> hulls;
    hulls.reserve(Z_.size());
    for (const Zonotope &A : Z_) hulls.push_back(A.enclose(M, around));
    return ZonoBundle(std::move(hulls));
}

Set ZonoBundle::enclose_impl(const ContSet &S2) const {
    // the entry of ContSet checked the dimension only: the members must pair up, checks on or off
    detail::Refusal r;
    if (S2.class_id() == ClassId::zono_bundle) {
        const ZonoBundle &S = static_cast<const ZonoBundle &>(S2);
        r = priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "enclose");
        if (!r) r = aux_count_refusal(*this, S);
        if (r) CORA_FAIL(r->first, r->second);
        return enclose(S);
    }
    if (S2.class_id() == ClassId::zonotope) {
        const Zonotope &Z = static_cast<const Zonotope &>(S2);
        r = priv_operand_refusal(*this, Z.dim(), Z.center(), "enclose");
        if (r) CORA_FAIL(r->first, r->second);
        return enclose(Z);
    }
    priv_noops("enclose", *this, &S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
