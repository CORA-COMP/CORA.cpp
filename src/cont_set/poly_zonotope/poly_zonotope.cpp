// poly_zonotope - the constructors of a polynomial zonotope, checked, as CORA's polyZonotope
//
// The constructor checks the shapes, merges the dependent generators with equal exponents and
// drops the generators that are zero (CORA's removeRedundantExponents and the zero columns). A
// zonotope or a box gives dependent generators with E = I.
//
// Syntax:   PolyZonotope pZ(c, G, GI, E, id);   PolyZonotope pZ(Z);   PolyZonotope pZ(I);
// Inputs:   c - center (n)
//           G - dependent generators (n, h)
//           GI - independent generators (n, q)
//           E - the exponents of the h generators over p factors
//           id - the p identifiers
// Outputs:  pZ - the polynomial zonotope
// See also: zonotope, exact_plus
//
// CORA: contSet/@polyZonotope/polyZonotope.m
// CORA: contSet/@zonotope/polyZonotope.m
// CORA: contSet/@interval/polyZonotope.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The exponent matrix I of h generators over h factors.
Exponents aux_identity(int64_t h) {
    Exponents E(std::size_t(h), std::vector<int>(std::size_t(h), 0));
    for (int64_t j = 0; j < h; ++j) E[std::size_t(j)][std::size_t(j)] = 1;
    return E;
}

/// The identifiers 1, ..., p.
std::vector<int64_t> aux_ids(int64_t p) {
    std::vector<int64_t> id(static_cast<std::size_t>(p), 0);
    for (int64_t i = 0; i < p; ++i) id[std::size_t(i)] = i + 1;
    return id;
}

/// The rows of GI that are not zero: a zero independent generator adds nothing.
Tensor aux_nonzero_rows(const Tensor &GI) {
    const int64_t q = GI.shape()[0], n = GI.shape()[1];
    const std::vector<double> g = GI.data();
    std::vector<int64_t> keep;
    for (int64_t j = 0; j < q; ++j)
        for (int64_t i = 0; i < n; ++i)
            if (g[std::size_t(j * n + i)] != 0.0) {
                keep.push_back(j);
                break;
            }
    return int64_t(keep.size()) == q ? GI : priv_rows(GI, keep);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The constructors --------------------------------------------------------------------------------

PolyZonotope::PolyZonotope(Tensor c, Tensor G, Tensor GI, Exponents E, std::vector<int64_t> id)
    : ContSet(kClassId), c_(std::move(c)), GI_(std::move(GI)), id_(std::move(id)) {
    detail::Entry entry;
    // the tensors, then the shapes of the center and of both generator matrices
    CORA_CHECK(c_.defined() && G.defined() && GI_.defined(), Error::wrong_input_in_constructor, [] {
        return std::string("PolyZonotope: the center and the generators must be tensors");
    });
    CORA_CHECK(c_.ndim() == 1, Error::wrong_input_in_constructor, [&] {
        return "PolyZonotope: the center must be a vector (n), got the shape " +
               shape_text(c_.shape());
    });
    const int64_t n = c_.shape()[0];
    // one generator per row: G (h, n), GI (q, n)
    CORA_CHECK(G.ndim() == 2 && G.shape()[1] == n && GI_.ndim() == 2 && GI_.shape()[1] == n,
               Error::wrong_input_in_constructor, [&] {
                   return "PolyZonotope: the generator rows need " + std::to_string(n) +
                          " entries (the dimension of the center), got the shapes " +
                          shape_text(G.shape()) + " and " + shape_text(GI_.shape());
               });
    // the exponents: one column per dependent generator, one entry per identifier, nonnegative
    CORA_CHECK(int64_t(E.size()) == G.shape()[0], Error::wrong_input_in_constructor, [&] {
        return "PolyZonotope: the exponent matrix needs one column per dependent generator (" +
               std::to_string(G.shape()[0]) + "), got " + std::to_string(E.size());
    });
    for (const std::vector<int> &col : E) {
        CORA_CHECK(col.size() == id_.size(), Error::wrong_input_in_constructor, [&] {
            return "PolyZonotope: every exponent column needs one entry per identifier (" +
                   std::to_string(id_.size()) + "), got " + std::to_string(col.size());
        });
        CORA_CHECK(std::all_of(col.begin(), col.end(), [](int e) { return e >= 0; }),
                   Error::wrong_input_in_constructor,
                   [] { return std::string("PolyZonotope: the exponents must be nonnegative"); });
    }
    // equal exponents add, zero generators go
    MergedColumns merged = priv_merge_columns(G, E);
    G_ = std::move(merged.G);
    E_ = std::move(merged.E);
    GI_ = aux_nonzero_rows(GI_);
}

PolyZonotope::PolyZonotope(Tensor c, Tensor G, const Tensor &GI, Exponents E,
                           std::vector<int64_t> id, detail::Normalized)
    : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)), GI_(aux_nonzero_rows(GI)),
      E_(std::move(E)), id_(std::move(id)) {
    // zero dependent generators go with their exponents
    const int64_t h = G_.shape()[0], n = G_.shape()[1];
    if (h == 0) return;
    const std::vector<double> g = G_.data();
    std::vector<int64_t> keep;
    for (int64_t j = 0; j < h; ++j)
        for (int64_t i = 0; i < n; ++i)
            if (g[std::size_t(j * n + i)] != 0.0) {
                keep.push_back(j);
                break;
            }
    if (int64_t(keep.size()) == h) return;
    Exponents kept;
    for (const int64_t j : keep) kept.push_back(E_[std::size_t(j)]);
    G_ = priv_rows(G_, keep);
    E_ = std::move(kept);
}

PolyZonotope::PolyZonotope(Tensor c, Tensor G, Tensor GI, Exponents E, std::vector<int64_t> id,
                           detail::Verbatim)
    : ContSet(kClassId), c_(std::move(c)), G_(std::move(G)), GI_(std::move(GI)), E_(std::move(E)),
      id_(std::move(id)) {}

// The conversions ---------------------------------------------------------------------------------

PolyZonotope::PolyZonotope(const Zonotope &Z)
    : PolyZonotope(Z.center(), Z.generators(), Z.center().zeros_like({0, Z.dim()}),
                   aux_identity(Z.nr_generators()), aux_ids(Z.nr_generators())) {}

PolyZonotope::PolyZonotope(const Interval &I) : PolyZonotope(I.zonotope()) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
