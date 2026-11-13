// enclose - a polynomial zonotope that contains the hull of two, as CORA's enclose
//
// With the same factors and exponents the hull is exact in the factors: the generators
// (G + G2)/2 and (G - G2)/2 and (c - c2)/2 on a new factor t in [-1, 1] give the points between the
// two sets at t. Otherwise the identifiers are merged and every generator [c, G] of each set is
// taken on t = 0 and t = 1, the constants joining the center. The independent generators are
// those of the enclosing zonotope.
//
// Syntax:   S = pZ.enclose(pZ2);
// Inputs:   pZ2 - a polynomial zonotope of the same dimension
// Outputs:  S - encloses the segments between the points of both
// See also: zonotope
//
// CORA: contSet/@polyZonotope/enclose.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The independent generators of the enclosure of the independent parts (CORA: zonotope enclose).
Tensor aux_independent(const PolyZonotope &A, const PolyZonotope &B) {
    const Tensor zero = A.center().zeros_like();
    const Zonotope Z1(zero, A.independent_generators()), Z2(zero, B.independent_generators());
    return Z1.enclose(Z2).generators();
}

/// The set with the independent generators as CORA's zonotope enclose gives them, the zero column
/// of the center difference included (the constructor drops it).
PolyZonotope aux_keep_independent(const PolyZonotope &body, const Tensor &GI) {
    return {body.center(), body.generators(), GI, body.exponents(), body.ids(), detail::Verbatim{}};
}

/// The matrix of columns with zeros below the p rows of E.
Exponents aux_with_row(const Exponents &E, int value, std::size_t count) {
    Exponents out;
    for (std::size_t j = 0; j < count; ++j) {
        std::vector<int> col = E[j];
        col.push_back(value);
        out.push_back(std::move(col));
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::enclose(const PolyZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "enclose: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    const int64_t new_id =
        (id_.empty() && S.id_.empty())
            ? 1
            : std::max(id_.empty() ? 0 : *std::max_element(id_.begin(), id_.end()),
                       S.id_.empty() ? 0 : *std::max_element(S.id_.begin(), S.id_.end())) +
                  1;
    const Tensor GI = aux_independent(*this, S);
    if (id_ == S.id_ && E_ == S.E_) {
        // the same factors: the points at t = 0 and 1, and the difference of the centers
        const Tensor G =
            Tensor::concat({G_ * 0.5 + S.G_ * 0.5, G_ * 0.5 - S.G_ * 0.5, (c_ - S.c_) * 0.5}, -1);
        const auto h = std::size_t(num_dependent());
        Exponents E = aux_with_row(E_, 0, h);
        for (const std::vector<int> &col : aux_with_row(E_, 1, h)) E.push_back(col);
        std::vector<int> last(id_.size() + 1, 0);
        last.back() = 1;
        E.push_back(std::move(last));
        std::vector<int64_t> id = id_;
        id.push_back(new_id);
        const PolyZonotope body((c_ + S.c_) * 0.5, G, GI, std::move(E), std::move(id));
        return aux_keep_independent(body, GI);
    }
    // other factors: [c, G] of each set on t = 0 and t = 1, over the union of the identifiers
    const MergedIds ids = priv_merge_ids(id_, S.id_);
    const std::size_t p = ids.id.size();
    Exponents E1 = priv_extend_exponents(E_, ids.position1, p);
    Exponents E2 = priv_extend_exponents(S.E_, ids.position2, p);
    E1.insert(E1.begin(), std::vector<int>(p, 0));
    E2.insert(E2.begin(), std::vector<int>(p, 0));
    const Tensor G1 = Tensor::concat({c_, G_}, -1), G2 = Tensor::concat({S.c_, S.G_}, -1);
    const Tensor G = Tensor::concat({G1, G1, G2, G2 * -1.0}, -1) * 0.5;
    Exponents E = aux_with_row(E1, 0, E1.size());
    for (const std::vector<int> &col : aux_with_row(E1, 1, E1.size())) E.push_back(col);
    for (const std::vector<int> &col : aux_with_row(E2, 0, E2.size())) E.push_back(col);
    for (const std::vector<int> &col : aux_with_row(E2, 1, E2.size())) E.push_back(col);
    std::vector<int64_t> id = ids.id;
    id.push_back(new_id);
    return aux_keep_independent(
        priv_fold_constants({c_.zeros_like(), G, GI, std::move(E), std::move(id)}), GI);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
