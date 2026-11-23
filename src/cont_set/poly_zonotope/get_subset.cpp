// get_subset - a polynomial zonotope with factors fixed or restricted, as CORA's getSubset
//
// A point value substitutes a factor (its power scales each generator that uses it). A box
// [c - r, c + r] of [-1, 1] re-parametrizes the factor by c + r a, a in [-1, 1]: each generator
// of exponent e in it becomes e + 1 generators with the terms of (c + r a)^e. A box that is
// [-1, 1] changes nothing. Afterwards equal exponents are added, the generators without exponent
// move to the center, and the factors that no generator uses go. CORA returns a vector for a set
// without generators, here it is a PolyZonotope with none.
//
// Syntax:   S = pZ.get_subset(ids, val);   S = pZ.get_subset(ids, I);
// Inputs:   ids - identifiers of the set
//           val - a value per identifier (n entries), or a box per identifier
// Outputs:  S - the subset
// See also: resolve, rm_id
//
// CORA: contSet/@polyZonotope/getSubset.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The generators and exponents while the factors are substituted one after the other.
struct State {
    Tensor G;
    Exponents E;
};

/// The position of the identifier, refused if the set lacks it.
std::size_t aux_position(const std::vector<int64_t> &ids, int64_t id) {
    const int64_t at = priv_find_id(ids, id);
    if (at < 0)
        CORA_FAIL(Error::wrong_value, "get_subset: ids must contain the identifiers of the "
                                      "factors that are changed");
    return std::size_t(at);
}

// The point ---------------------------------------------------------------------------------------

/// The factor at `at` set to the value v (1 entry): every generator that uses it is scaled by v^e.
void aux_point(State &s, std::size_t at, const Tensor &v) {
    std::vector<double> exponent;
    bool any = false;
    for (std::vector<int> &col : s.E) {
        exponent.push_back(col[at]);
        any = any || col[at] != 0;
        col[at] = 0;
    }
    if (!any) return;
    const Tensor scale = v.pow(Tensor::like(s.G, exponent, {int64_t(exponent.size())}));
    s.G = s.G.multiply(scale.expand_dims(-1));
}

// The box -----------------------------------------------------------------------------------------

/// The coefficients of (b a + c)^e by the powers of a, highest first, as CORA's conv of [b c].
std::vector<std::vector<Tensor>> aux_coefficients(const Tensor &c, const Tensor &b, int e_max) {
    std::vector<std::vector<Tensor>> coeffs = {{}, {b, c}};
    for (int e = 2; e <= e_max; ++e) {
        const std::vector<Tensor> &u = coeffs.back();
        std::vector<Tensor> y;
        for (std::size_t k = 0; k <= u.size(); ++k) {
            if (k == 0)
                y.push_back(u[0].multiply(b));
            else if (k == u.size())
                y.push_back(u[k - 1].multiply(c));
            else
                y.push_back(u[k - 1].multiply(c) + u[k].multiply(b));
        }
        coeffs.push_back(std::move(y));
    }
    return coeffs;
}

/// The factor at `at` replaced by c + r a: the binomial expansion of every generator that uses it.
void aux_box(State &s, std::size_t at, const Tensor &c, const Tensor &r) {
    int e_max = 0;
    for (const std::vector<int> &col : s.E) e_max = std::max(e_max, col[at]);
    if (e_max == 0) return;
    const std::vector<std::vector<Tensor>> coeffs = aux_coefficients(c, r, e_max);
    // a generator that uses the factor: the constant term stays, one column per other term
    std::vector<Tensor> scale;
    std::vector<int64_t> source;
    std::vector<Tensor> coefficient;
    Exponents appended;
    for (std::size_t j = 0; j < s.E.size(); ++j) {
        const int e = s.E[j][at];
        scale.push_back(e == 0 ? c.ones_like() : coeffs[std::size_t(e)].back());
        for (int t = 0; t < e; ++t) {
            source.push_back(int64_t(j));
            coefficient.push_back(coeffs[std::size_t(e)][std::size_t(t)]);
            appended.push_back(s.E[j]);
            appended.back()[at] = e - t;
        }
        s.E[j][at] = 0;
    }
    // the old columns scaled by the constant term, the new ones appended
    Tensor G = s.G.multiply(Tensor::concat(scale, 0).expand_dims(-1));
    if (!source.empty()) {
        const Tensor more =
            priv_rows(s.G, source).multiply(Tensor::concat(coefficient, 0).expand_dims(-1));
        G = Tensor::concat({G, more}, -2);
        s.E.insert(s.E.end(), appended.begin(), appended.end());
    }
    s.G = std::move(G);
}

/// The ending of CORA's getSubset.
PolyZonotope aux_finish(const PolyZonotope &set, State &s) {
    return priv_merged_set(set.center(), s.G, set.independent_generators(), s.E, set.ids());
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::get_subset(const std::vector<int64_t> &ids, const Tensor &val) const {
    detail::Entry entry;
    CORA_CHECK(val.ndim() == 1 && val.shape()[0] >= int64_t(ids.size()), Error::wrong_value,
               [] { return std::string("get_subset: val needs one value per identifier"); });
    State s{G_, E_};
    // one factor after the other, as CORA does
    for (std::size_t i = 0; i < ids.size(); ++i)
        aux_point(s, aux_position(id_, ids[i]), val.slice(-1, int64_t(i), int64_t(i) + 1));
    return aux_finish(*this, s);
}

PolyZonotope PolyZonotope::get_subset(const std::vector<int64_t> &ids, const Interval &val) const {
    detail::Entry entry;
    CORA_CHECK(!val.is_empty(), Error::wrong_value,
               [] { return std::string("get_subset: val should not be empty"); });
    CORA_CHECK(val.inf().ndim() == 1 && val.inf().shape()[0] >= int64_t(ids.size()),
               Error::wrong_value,
               [] { return std::string("get_subset: val needs one box per identifier"); });
    // the center and the radius of every box
    const Tensor center = val.center(), radius = (val.sup() - val.inf()) * 0.5;
    State s{G_, E_};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Tensor c = center.slice(-1, int64_t(i), int64_t(i) + 1),
                     r = radius.slice(-1, int64_t(i), int64_t(i) + 1);
        // the factor ranging over [-1, 1] itself stays
        if (r.item() == 1.0 && c.item() == 0.0) continue;
        aux_box(s, aux_position(id_, ids[i]), c, r);
    }
    return aux_finish(*this, s);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
