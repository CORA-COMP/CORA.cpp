// isequal - whether two polynomial zonotopes are equal, as CORA's isequal
//
// Both sets are compacted, their identifiers must be the same and are aligned, and the numbers of
// generators agree. The dependent generators with their exponents are compared up to order, the
// independent generators up to order and sign, the centers and all entries within tol. Equal data
// denote one set; the converse need not hold. A zonotope or a box is compared as a polynomial
// zonotope.
//
// Syntax:   res = pZ.isequal(S);   res = pZ.isequal(S, tol);
// Inputs:   S - a polynomial zonotope, a zonotope or a box
//           tol - the tolerance of the comparison; default: the repository tolerance
// Outputs:  res - true if the data agree; a different dimension gives false
// See also: compact, exact_plus
//
// CORA: contSet/@polyZonotope/isequal.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the rows of A and of B (m rows of n entries) agree up to order, each row of A taking the
/// first unused row of B within tol (with the opposite sign too if `signed_` is false): CORA's
/// compareMatrices on columns. The exponents are the last `extra` entries of a row.
bool aux_same_rows(const std::vector<std::vector<double>> &A,
                   const std::vector<std::vector<double>> &B, bool signed_, double tol) {
    std::vector<bool> used(B.size(), false);
    const auto agree = [&](const std::vector<double> &a, const std::vector<double> &b, double s) {
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!within_tol(a[i], s * b[i], tol)) return false;
        return true;
    };
    for (const std::vector<double> &a : A) {
        bool found = false;
        for (std::size_t j = 0; j < B.size() && !found; ++j)
            if (!used[j] && (agree(a, B[j], 1.0) || (!signed_ && agree(a, B[j], -1.0))))
                found = used[j] = true;
        if (!found) return false;
    }
    return true;
}

// The generators ---------------------------------------------------------------------------------

/// The rows of G with the exponents (reordered to the identifiers `position` of `rows`) appended.
std::vector<std::vector<double>>
aux_dependent(const PolyZonotope &P, const std::vector<std::size_t> &position, std::size_t rows) {
    // the exponents over the common identifiers, appended to each generator
    const Exponents E = priv_extend_exponents(P.exponents(), position, rows);
    const int64_t n = P.dim();
    const std::vector<double> g = P.generators().data();
    std::vector<std::vector<double>> out;
    // each row: the generator, then its exponents
    for (std::size_t j = 0; j < E.size(); ++j) {
        std::vector<double> row(g.begin() + std::ptrdiff_t(std::size_t(n) * j),
                                g.begin() + std::ptrdiff_t(std::size_t(n) * (j + 1)));
        row.insert(row.end(), E[j].begin(), E[j].end());
        out.push_back(std::move(row));
    }
    return out;
}

std::vector<std::vector<double>> aux_independent(const PolyZonotope &P) {
    // one row of n entries per independent generator
    const int64_t n = P.dim();
    const std::vector<double> g = P.independent_generators().data();
    std::vector<std::vector<double>> out;
    for (int64_t j = 0; j < P.independent_generators().shape()[0]; ++j)
        out.emplace_back(g.begin() + std::ptrdiff_t(j * n),
                         g.begin() + std::ptrdiff_t((j + 1) * n));
    return out;
}

// The comparison ---------------------------------------------------------------------------------

/// Whether the polynomial zonotopes are equal up to tol.
bool aux_isequal(const PolyZonotope &P, const PolyZonotope &S, double tol) {
    if (S.dim() != P.dim()) return false;
    const PolyZonotope a = P.compact(), b = S.compact();
    if (a.num_dependent() != b.num_dependent() ||
        a.independent_generators().shape()[0] != b.independent_generators().shape()[0])
        return false;
    std::vector<int64_t> ia = a.ids(), ib = b.ids();
    std::sort(ia.begin(), ia.end());
    std::sort(ib.begin(), ib.end());
    if (ia != ib) return false;
    // the same identifiers, in the order of their union
    const MergedIds ids = priv_merge_ids(a.ids(), b.ids());
    if (!aux_same_rows(aux_dependent(a, ids.position1, ids.id.size()),
                       aux_dependent(b, ids.position2, ids.id.size()), true, tol) ||
        !aux_same_rows(aux_independent(a), aux_independent(b), false, tol))
        return false;
    // the centers
    const std::vector<double> ca = a.center().data(), cb = b.center().data();
    for (std::size_t i = 0; i < ca.size(); ++i)
        if (!within_tol(ca[i], cb[i], tol)) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool PolyZonotope::isequal(const PolyZonotope &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "isequal");
    return aux_isequal(*this, S, tol);
}

bool PolyZonotope::isequal(const Zonotope &Z, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "isequal");
    return aux_isequal(*this, PolyZonotope(Z), tol);
}

bool PolyZonotope::isequal(const Interval &I, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "isequal");
    return !I.is_empty() && aux_isequal(*this, PolyZonotope(I), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
