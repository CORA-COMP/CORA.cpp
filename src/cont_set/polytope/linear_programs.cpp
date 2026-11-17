// linear_programs - the linear programs over a polytope cut by a box, prepared for many costs
//
// The box stays the bounds lb <= x <= ub of every program, as CORA hands the unit box of a
// constrained zonotope to CORAlinprog; the rows are the halfspaces of the polytope. Without
// variables the programs read 0 <= b and 0 = be and are decided up to the simplex tolerance.
//
// Syntax:   auto lp = P.linear_programs(lb, ub);   bool empty = lp.is_empty();
//           auto r = lp.range(f);   auto best = lp.maximum(f);   auto [lo, hi] = lp.bounds();
//           auto x = lp.chebyshev_center();
// Inputs:   P - a single polytope
//           lb, ub - the box (n values each)
//           f - a cost (n values)
// Outputs:  R - emptiness, the range or maximum of f'x, the coordinate ranges, or the center
// See also: ConZonotope, linprog
//
// CORA.cpp only: CORA's conZonotope calls CORAlinprog with its factor box inline
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the box has n bounds on each side, and the refusal if not.
bool aux_box_fits(const std::vector<double> &lb, const std::vector<double> &ub, int64_t n) {
    return int64_t(lb.size()) == n && int64_t(ub.size()) == n;
}
std::string aux_box_text(const std::vector<double> &lb, const std::vector<double> &ub, int64_t n) {
    return "linear_programs: the box needs " + std::to_string(n) + " bounds per side, got " +
           std::to_string(lb.size()) + " and " + std::to_string(ub.size());
}

/// The values of a (rows, cols) matrix on the host, row-major; none for zero rows.
std::vector<double> aux_host_rows(const Tensor &T) {
    return T.shape()[0] == 0 ? std::vector<double>{} : T.data();
}

/// The program without variables: the rows read 0 <= b and 0 = be, which hold up to the
/// feasibility tolerance of the simplex method; the optimum is 0.
LinProgResult aux_no_variables(const LinProg &lp) {
    const auto small = [](double x) { return std::abs(x) <= numerics::kSimplexFeasible; };
    const bool feasible = std::all_of(lp.beq.begin(), lp.beq.end(), small) &&
                          std::all_of(lp.bineq.begin(), lp.bineq.end(),
                                      [](double x) { return x >= -numerics::kSimplexFeasible; });
    LinProgResult none;
    none.status = feasible ? LinProgResult::Status::optimal : LinProgResult::Status::infeasible;
    none.fval = feasible ? 0.0 : none.fval;
    return none;
}

/// The program with each of the costs (they share phase 1).
std::vector<LinProgResult> aux_solve(const LinProg &lp,
                                     const std::vector<std::vector<double>> &costs) {
    if (lp.n > 0) return linprog_costs(lp, costs);
    return std::vector<LinProgResult>(costs.size(), aux_no_variables(lp));
}

/// The program with the one cost f.
LinProgResult aux_solve(LinProg lp, const std::vector<double> &f) {
    if (lp.n == 0) return aux_no_variables(lp);
    lp.f = f;
    return linprog(lp);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The programs ------------------------------------------------------------------------------------

Polytope::LinearPrograms::LinearPrograms(const Polytope &P, std::vector<double> lb,
                                         std::vector<double> ub) {
    detail::Entry entry;
    // the programs run on one set, checked even inside another operation
    if (!P.is_h_rep() || P.A().ndim() != 2)
        CORA_FAIL(Error::not_supported,
                  "LinearPrograms: a single polytope in halfspace form, not a batch");
    const int64_t n = P.dim();
    CORA_CHECK(aux_box_fits(lb, ub, n), Error::dimension_mismatch,
               [&] { return aux_box_text(lb, ub, n); });
    // the rows of the polytope, the box as bounds
    lp_.n = int(n);
    lp_.Aineq = aux_host_rows(P.A());
    lp_.bineq = aux_host_rows(P.b());
    lp_.Aeq = aux_host_rows(P.Ae());
    lp_.beq = aux_host_rows(P.be());
    lp_.lb = std::move(lb);
    lp_.ub = std::move(ub);
}

Polytope::LinearPrograms Polytope::linear_programs(std::vector<double> lb,
                                                   std::vector<double> ub) const {
    detail::Entry entry;
    CORA_CHECK(aux_box_fits(lb, ub, dim()), Error::dimension_mismatch,
               [&] { return aux_box_text(lb, ub, dim()); });
    return {*this, std::move(lb), std::move(ub)};
}

// Emptiness ---------------------------------------------------------------------------------------

bool Polytope::LinearPrograms::is_empty() const {
    detail::Entry entry;
    // the box alone has a point
    if (lp_.bineq.empty() && lp_.beq.empty()) return false;
    return aux_solve(lp_, std::vector<double>(std::size_t(lp_.n), 0.0)).status !=
           LinProgResult::Status::optimal;
}

// Ranges and maxima -------------------------------------------------------------------------------

std::optional<std::pair<double, double>>
Polytope::LinearPrograms::range(const std::vector<double> &f) const {
    detail::Entry entry;
    // both ends from the programs of f and -f
    std::vector<double> neg = f;
    for (double &x : neg) x = -x;
    const std::vector<LinProgResult> both = aux_solve(lp_, {f, neg});
    const LinProgResult &lower = both[0], &upper = both[1];
    if (lower.status != LinProgResult::Status::optimal) return std::nullopt;
    if (upper.status != LinProgResult::Status::optimal) return std::nullopt;
    return std::pair{lower.fval, -upper.fval};
}

std::optional<Polytope::LinearPrograms::Maximum>
Polytope::LinearPrograms::maximum(const std::vector<double> &f) const {
    detail::Entry entry;
    // the maximum of f is the negated minimum of -f
    std::vector<double> neg = f;
    for (double &x : neg) x = -x;
    LinProgResult best = aux_solve(lp_, neg);
    if (best.status != LinProgResult::Status::optimal) return std::nullopt;
    return Maximum{-best.fval, std::move(best.x)};
}

std::pair<std::vector<double>, std::vector<double>> Polytope::LinearPrograms::bounds() const {
    detail::Entry entry;
    const std::size_t n = std::size_t(lp_.n);
    // the unit costs of both ends of every coordinate, solved together
    std::vector<std::vector<double>> costs;
    costs.reserve(2 * n);
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> f(n, 0.0);
        f[j] = 1.0;
        costs.push_back(f);
        f[j] = -1.0;
        costs.push_back(f);
    }
    const std::vector<LinProgResult> solved = aux_solve(lp_, costs);
    std::vector<double> lo(n), hi(n);
    for (std::size_t j = 0; j < n; ++j) {
        lo[j] = solved[2 * j].fval;
        hi[j] = -solved[2 * j + 1].fval;
    }
    return {lo, hi};
}

// The Chebyshev center ----------------------------------------------------------------------------

std::optional<std::vector<double>> Polytope::LinearPrograms::chebyshev_center() const {
    detail::Entry entry;
    const std::size_t n = std::size_t(lp_.n);
    HostH H;
    H.n = lp_.n;
    // the halfspaces, then the finite lower and upper sides of the box
    for (std::size_t i = 0; i < lp_.bineq.size(); ++i)
        H.A.emplace_back(lp_.Aineq.begin() + std::ptrdiff_t(i * n),
                         lp_.Aineq.begin() + std::ptrdiff_t((i + 1) * n));
    H.b = lp_.bineq;
    for (const double sign : {-1.0, 1.0}) {
        const std::vector<double> &side = sign < 0 ? lp_.lb : lp_.ub;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isinf(side[j])) continue;
            Row row(n, 0.0);
            row[j] = sign;
            H.A.push_back(row);
            H.b.push_back(sign * side[j]);
        }
    }
    for (std::size_t i = 0; i < lp_.beq.size(); ++i)
        H.Ae.emplace_back(lp_.Aeq.begin() + std::ptrdiff_t(i * n),
                          lp_.Aeq.begin() + std::ptrdiff_t((i + 1) * n));
    H.be = lp_.beq;
    return priv_chebyshev_center(H);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
