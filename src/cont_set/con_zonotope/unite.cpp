// unite - a constrained zonotope that encloses the union of constrained zonotopes, as CORA's or
//
// The sets with constraints are rescaled and their zonotopes enclosed pairwise (the iterative
// union); the generators of that enclosure, of unit length, are scaled and its center moved, by a
// linear program, to the least that still cover every set along the facet normals of the
// enclosure (the support function of each set is a linear program). "tedrake" fails in CORA (it
// uses a variable it never sets) and is Error::not_supported, as is an order reduction (order).
// Not a batch.
//
// Syntax:   cZu = cz.unite(S);   cZu = cz.unite({S2, S3}, "linprog");   cZu = cz.unite(p);
// Inputs:   S - a constrained zonotope, a zonotope, a box, a point (n, 1), or a list of them
//           method - "linprog" (default)
// Outputs:  cZu - a set without constraints that contains the union
// See also: rescale, Zonotope::unite
//
// CORA: contSet/@conZonotope/or.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Checks ------------------------------------------------------------------------------------------

/// Why the union is refused, if it is: the method or the order (tedrake fails in CORA), a batch,
/// or another batch shape or dimension.
detail::Refusal aux_refusal(const ConZonotope &cz, const std::vector<ConZonotope> &others,
                            const std::string &method, std::optional<int> order) {
    if (method != "linprog" && method != "tedrake")
        return std::pair{Error::wrong_value,
                         "unite: the method must be 'linprog' or 'tedrake', got '" + method + "'"};
    if (method == "tedrake")
        return std::pair{Error::not_supported,
                         std::string("unite: the method 'tedrake' fails in CORA and is not "
                                     "supported")};
    if (order)
        return std::pair{Error::not_supported,
                         std::string("unite: reducing the order needs the reduction methods "
                                     "'methC' and 'pca', which are not supported")};
    if (cz.c().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("unite: it supports single sets, not a batch")};
    for (const ConZonotope &S : others)
        if (const detail::Refusal r = priv_pair_refusal("unite", cz, S, true)) return r;
    return std::nullopt;
}

// The union ---------------------------------------------------------------------------------------

/// The union of cz and the sets (all valid) by "linprog".
ConZonotope aux_unite(const ConZonotope &cz, const std::vector<ConZonotope> &others) {
    std::vector<ConZonotope> sets{cz};
    sets.insert(sets.end(), others.begin(), others.end());

    // the zonotopes of the rescaled sets, enclosed pairwise
    std::vector<Zonotope> zonotopes;
    for (const ConZonotope &S : sets) {
        const ConZonotope r = S.A().numel() == 0 ? S : S.rescale();
        zonotopes.emplace_back(r.c(), r.generators());
    }
    const Zonotope iterative = zonotopes.front().unite(
        std::vector<Zonotope>(zonotopes.begin() + 1, zonotopes.end()), "iterative");

    // the support value of every set along every row of A, and the largest of them
    std::vector<HostConZonotope> hosts;
    hosts.reserve(sets.size());
    for (const ConZonotope &S : sets) hosts.push_back(priv_host_con_zonotope(S));
    const auto support = [&](const Tensor &A) {
        const int64_t k = A.shape()[0], n = A.shape()[1];
        const std::vector<double> rows = A.data();
        std::vector<double> d(std::size_t(k), -std::numeric_limits<double>::infinity());
        for (int64_t i = 0; i < k; ++i)
            for (const HostConZonotope &H : hosts) {
                const auto range = priv_range(
                    H, std::vector<double>(rows.begin() + i * n, rows.begin() + (i + 1) * n));
                if (range) d[i] = std::max(d[i], range->second);
            }
        return Tensor::like(A, d, {k, 1});
    };
    const Zonotope Z = priv_union_linprog(iterative, support);
    return {Z.center(), Z.generators()};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Overloads ---------------------------------------------------------------------------------------
ConZonotope ConZonotope::unite(const std::vector<ConZonotope> &others, const std::string &method,
                               std::optional<int> order) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, others, method, order);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_unite(*this, others);
}

ConZonotope ConZonotope::unite(const ConZonotope &S, const std::string &method,
                               std::optional<int> order) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, {S}, method, order);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_unite(*this, {S});
}

// a zonotope is a set without constraints
ConZonotope ConZonotope::unite(const Zonotope &Z, const std::string &method,
                               std::optional<int> order) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, {ConZonotope(Z)}, method, order);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_unite(*this, {ConZonotope(Z)});
}

// and so is a box
ConZonotope ConZonotope::unite(const Interval &I, const std::string &method,
                               std::optional<int> order) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, {ConZonotope(I)}, method, order);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_unite(*this, {ConZonotope(I)});
}

// a point is a set without generators
ConZonotope ConZonotope::unite(const Tensor &p, const std::string &method,
                               std::optional<int> order) const {
    detail::Entry entry;
    const ConZonotope S(p, priv_zeros(p, p.shape()[p.ndim() - 2], 0));
    const detail::Refusal refusal = aux_refusal(*this, {S}, method, order);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_unite(*this, {S});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
