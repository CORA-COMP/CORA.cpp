// enclose - a constrained zonotope that encloses a set and its image under an affine map
//
// Syntax:   cZres = cz.enclose(cz2);   cZres = cz.enclose(M, S);   cZres = cz.enclose(M, t);
// Inputs:   cz2 - the second set, which must be a linear image of cz: its constraints are those of
//           cz (the first columns of A and b agree up to eps); M, S - the second set is M cz + S
//           for a constrained zonotope S, or M cz + t for a point t (n2, 1)
// Outputs:  cZres - encloses the union: center (c+c2)/2; the generators in common give (G+G2)/2
//           and (G-G2)/2, then come (c-c2)/2 and the extra generators of the longer set; the
//           constraints are those of cz on the first factors, none on the new ones
// See also: convex_hull, Zonotope::enclose
//
// CORA: contSet/@conZonotope/enclose.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why cz2 is not an image of cz, if it is not: the constraints must agree on the common factors,
/// and the factors that only the longer set has must not appear in its constraints (CORA drops
/// them, which would not enclose).
std::string aux_problem(const ConZonotope &cz, const ConZonotope &cz2) {
    const int64_t g1 = cz.generators().shape().back(), g2 = cz2.generators().shape().back();
    const int64_t g = std::min(g1, g2), p = cz.A().shape()[cz.A().ndim() - 2];
    const int64_t p2 = cz2.A().shape()[cz2.A().ndim() - 2];
    const std::string image =
        "enclose: only valid if the second set is a linear image of the first";
    if (p != p2) return image;
    if (p == 0) return "";
    const Tensor A1 = cz.A().slice(-1, 0, g), A2 = cz2.A().slice(-1, 0, g);
    const double eps = machine_eps(cz.c().dtype());
    if ((A1 - A2).abs().max().item() > eps || (cz.b() - cz2.b()).abs().max().item() > eps)
        return image;
    // the longer set's constraints beyond the common factors
    const ConZonotope &longer = g2 <= g1 ? cz : cz2;
    const Tensor rest = longer.A().slice(-1, g, longer.A().shape().back());
    if (rest.shape().back() > 0 && rest.abs().max().item() > 0)
        return "enclose: the constraints of the longer set use factors that the other lacks";
    return "";
}

/// Why the second set is refused, if it is: another dimension, or no image of the first set.
detail::Refusal aux_refusal(const ConZonotope &cz, const ConZonotope &cz2) {
    if (cz2.dim() != cz.dim())
        return std::pair{Error::dimension_mismatch, "enclose: the sets differ in dimension, " +
                                                        std::to_string(cz.dim()) + " and " +
                                                        std::to_string(cz2.dim())};
    const std::string problem = aux_problem(cz, cz2);
    if (problem.empty()) return std::nullopt;
    return std::pair{Error::special_error, problem};
}

/// The enclosure: the zonotope part is that of Zonotope::enclose, the constraints those of the
/// set with fewer generators (they agree on the common factors), padded with zeros.
ConZonotope aux_enclosure(const ConZonotope &cz, const ConZonotope &cz2) {
    const Zonotope Z =
        Zonotope(cz.c(), cz.generators()).enclose(Zonotope(cz2.c(), cz2.generators()));
    const bool second_shorter = cz2.generators().shape().back() <= cz.generators().shape().back();
    const Tensor &A = second_shorter ? cz2.A() : cz.A();
    const int64_t more = Z.generators().shape().back() - A.shape().back();
    const Tensor padded = Tensor::concat({A, priv_zeros(A, A.shape()[A.ndim() - 2], more)}, -1);
    return {Z.center(), Z.generators(), padded, cz.b()};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::enclose(const ConZonotope &cz2) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, cz2);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_enclosure(*this, cz2);
}

ConZonotope ConZonotope::enclose(const Tensor &M, const ConZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.shape().back() == dim() && M.shape()[M.ndim() - 2] == S.dim(),
               Error::dimension_mismatch, [&] {
                   return "enclose: M must be (" + std::to_string(S.dim()) + ", " +
                          std::to_string(dim()) + "), got the shape " + shape_text(M.shape());
               });
    // the second set is M this + S
    const ConZonotope image = priv_minkowski_sum(priv_affine_image(*this, M, std::nullopt), S);
    const detail::Refusal refusal = aux_refusal(*this, image);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_enclosure(*this, image);
}

ConZonotope ConZonotope::enclose(const Tensor &M, const Tensor &t) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.shape().back() == dim() && is_column(t, M.shape()[M.ndim() - 2]),
               Error::dimension_mismatch, [&] {
                   return "enclose: M must be (k, " + std::to_string(dim()) +
                          ") and t a column (k, 1), got the shapes " + shape_text(M.shape()) +
                          " and " + shape_text(t.shape());
               });
    const ConZonotope image = priv_affine_image(*this, M, t);
    const detail::Refusal refusal = aux_refusal(*this, image);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_enclosure(*this, image);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
