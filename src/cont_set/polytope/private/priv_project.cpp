// priv_project - the projection and the Minkowski sum of halfspace polytopes, on the host
//
// The projection eliminates the other coordinates one by one (Fourier-Motzkin), normalizing and
// compacting the rows after every step, as CORA's fallback 'fourier' does. The sum of two sets is
// the projection of their product: s = x + y with x in the first set and y in the second.
//
// Syntax:   auto Hp = priv_project_h(H, dims);   auto Hs = priv_plus_h(H1, H2);
// Inputs:   H, H1, H2 - single polytopes in halfspace form
//           dims - the coordinates to keep (0-based, in the order of the result)
// Outputs:  Hp, Hs - the projection and the sum, or none if the set is empty
// See also: priv_compact, Polytope::project, Polytope::plus
//
// CORA: contSet/@polytope/project.m
// CORA: contSet/@polytope/plus.m
// CORA authors: Mark Wetzlinger, Viktor Kotsev
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Elimination -------------------------------------------------------------------------------------

/// The row without its entry j.
Row aux_without(const Row &row, std::size_t j) {
    Row out = row;
    out.erase(out.begin() + std::ptrdiff_t(j));
    return out;
}

/// One Fourier-Motzkin step: the rows with column j eliminated (CORA's
/// aux_fourierMotzkinElimination): rows without the coordinate stay, and every pair of a row that
/// bounds it from below and one from above gives the row without it.
void aux_eliminate(Rows &A, Row &b, std::size_t j) {
    std::vector<std::size_t> zero, negative, positive;
    for (std::size_t i = 0; i < A.size(); ++i)
        (A[i][j] == 0.0 ? zero : A[i][j] < 0.0 ? negative : positive).push_back(i);
    Rows rows;
    Row offsets;
    for (const std::size_t i : zero) {
        rows.push_back(aux_without(A[i], j));
        offsets.push_back(b[i]);
    }
    // the pair (n, p): the positive multiple of n by A(p, j) plus that of p by -A(n, j)
    for (const std::size_t n : negative) {
        for (const std::size_t p : positive) {
            Row row(A[n].size());
            for (std::size_t k = 0; k < row.size(); ++k)
                row[k] = A[p][j] * A[n][k] - A[n][j] * A[p][k];
            rows.push_back(aux_without(row, j));
            offsets.push_back(A[p][j] * b[n] - A[n][j] * b[p]);
        }
    }
    A = rows;
    b = offsets;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<HostH> priv_project_h(HostH H, const std::vector<int64_t> &dims) {
    // the compaction finds the empty set; a set that is not found empty is projected
    priv_normalize_rows(H);
    bool minimal = false;
    if (priv_compact_all(H, numerics::kPolytopeProject, minimal)) return std::nullopt;
    auto [A, b] = priv_equality_to_inequality(H);
    // the columns that stay, in increasing order; the others go one by one
    std::vector<int64_t> columns(std::size_t(H.n));
    std::iota(columns.begin(), columns.end(), int64_t{0});
    for (int64_t d = 0; d < H.n; ++d) {
        if (std::find(dims.begin(), dims.end(), d) != dims.end()) continue;
        const auto position = std::find(columns.begin(), columns.end(), d);
        aux_eliminate(A, b, std::size_t(position - columns.begin()));
        columns.erase(position);
        // keep the rows few: normalize and compact (the result of an empty step is empty)
        HostH step{int64_t(columns.size()), A, b, {}, {}};
        priv_normalize_rows(step);
        if (priv_compact_all(step, numerics::kPolytopeProject, minimal)) return std::nullopt;
        A = step.A;
        b = step.b;
    }
    // the column of the k-th smallest coordinate goes to its place in dims
    HostH out{int64_t(dims.size()), Rows(A.size(), Row(dims.size())), b, {}, {}};
    for (std::size_t k = 0; k < columns.size(); ++k) {
        const auto place =
            std::size_t(std::find(dims.begin(), dims.end(), columns[k]) - dims.begin());
        for (std::size_t i = 0; i < A.size(); ++i) out.A[i][place] = A[i][k];
    }
    return out;
}

// The sum -----------------------------------------------------------------------------------------

std::optional<HostH> priv_plus_h(const HostH &H1, const HostH &H2) {
    const std::size_t n = std::size_t(H1.n);
    // the rows of (s, x): s - x in the second set, x in the first
    const auto join = [&](const Rows &first, const Rows &second) {
        Rows rows;
        for (const Row &a : second) {
            Row row(a);
            for (const double x : a) row.push_back(-x);
            rows.push_back(row);
        }
        for (const Row &a : first) {
            Row row(n, 0.0);
            row.insert(row.end(), a.begin(), a.end());
            rows.push_back(row);
        }
        return rows;
    };
    // the product of both sets in (s, x), projected onto s
    HostH both{2 * H1.n, join(H1.A, H2.A), H2.b, join(H1.Ae, H2.Ae), H2.be};
    both.b.insert(both.b.end(), H1.b.begin(), H1.b.end());
    both.be.insert(both.be.end(), H1.be.begin(), H1.be.end());
    std::vector<int64_t> keep(n);
    std::iota(keep.begin(), keep.end(), int64_t{0});
    return priv_project_h(both, keep);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
