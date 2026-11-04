// enclose - a zonotope bundle that encloses a bundle and a linear image of it, as CORA's enclose
//
// Syntax:   ZonoBundle R = zB.enclose(S);   ZonoBundle R = zB.enclose(M, S);
// Inputs:   S - a bundle with as many members, or a zonotope (every member is enclosed with it);
//           M - a square matrix (..., n, n): the second set is M zB + S, for a bundle S of any size
// Outputs:  R - the members enclosed one by one (Zonotope::enclose): member i of M zB + S is
//           M Z_i + S for a zonotope S, and M Z_i + the zonotope of the box around S for a bundle S
//           (CORA's plus of a bundle with a bundle); an enclosure of the convex hull of the union
// See also: Zonotope::enclose, intersect
//
// CORA: contSet/@zonoBundle/enclose.m

#include "cont_set/cont_set/private/priv_noops.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why M is refused, if it is: it is not (..., n, n) of the dimension of the bundle.
detail::Refusal aux_matrix_refusal(const ZonoBundle &zB, const Tensor &M) {
    if (M.ndim() >= 2 && M.size(-2) == zB.dim() && M.size(-1) == zB.dim()) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "enclose: M must be square of the dimension " + std::to_string(zB.dim()) +
                         " of the bundle, got the shape " + shape_text(M.shape())};
}

/// Why a bundle S with as many members is required, if it is not given.
detail::Refusal aux_count_refusal(const ZonoBundle &zB, const ZonoBundle &S) {
    if (S.parallel_sets() == zB.parallel_sets()) return std::nullopt;
    return std::pair{Error::wrong_value,
                     "enclose: the bundles must have the same number of zonotopes, got " +
                         std::to_string(zB.parallel_sets()) + " and " +
                         std::to_string(S.parallel_sets())};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::enclose(const ZonoBundle &S) const {
    detail::Entry entry;
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "enclose");
    if (!r) r = aux_count_refusal(*this, S);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    std::vector<Zonotope> hulls;
    for (std::size_t i = 0; i < Z_.size(); ++i) hulls.push_back(Z_[i].enclose(S.zonotopes()[i]));
    return ZonoBundle(std::move(hulls));
}

ZonoBundle ZonoBundle::enclose(const Zonotope &Z) const {
    detail::Entry entry;
    const detail::Refusal r = priv_operand_refusal(*this, Z.dim(), Z.center(), "enclose");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    std::vector<Zonotope> hulls;
    for (const Zonotope &A : Z_) hulls.push_back(A.enclose(Z));
    return ZonoBundle(std::move(hulls));
}

ZonoBundle ZonoBundle::enclose(const Tensor &M, const Zonotope &Zplus) const {
    detail::Entry entry;
    detail::Refusal r = priv_operand_refusal(*this, Zplus.dim(), Zplus.center(), "enclose");
    if (!r) r = aux_matrix_refusal(*this, M);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    std::vector<Zonotope> hulls;
    for (const Zonotope &A : Z_) hulls.push_back(A.enclose(M, Zplus));
    return ZonoBundle(std::move(hulls));
}

ZonoBundle ZonoBundle::enclose(const Tensor &M, const ZonoBundle &S) const {
    detail::Entry entry;
    detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "enclose");
    if (!r) r = aux_matrix_refusal(*this, M);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    const Interval box = S.interval();
    CORA_CHECK(!box.is_empty(), Error::not_supported,
               [] { return std::string("enclose: the second bundle is empty"); });
    const Zonotope around = box.zonotope();
    std::vector<Zonotope> hulls;
    for (const Zonotope &A : Z_) hulls.push_back(A.enclose(M, around));
    return ZonoBundle(std::move(hulls));
}

Set ZonoBundle::enclose_impl(const ContSet &S2) const {
    if (S2.class_id() == ClassId::zono_bundle) return enclose(static_cast<const ZonoBundle &>(S2));
    if (S2.class_id() == ClassId::zonotope) return enclose(static_cast<const Zonotope &>(S2));
    priv_noops("enclose", *this, &S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
