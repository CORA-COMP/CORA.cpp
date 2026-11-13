// linprog - a dense linear program solver (two-phase simplex), as CORA's CORAlinprog
//
// Solves  min f'x  s.t.  Aineq x <= bineq,  Aeq x = beq,  lb <= x <= ub  for the small programs
// of set computations (a few hundred variables); the matrices are row-major on the host.
//
// Syntax:     LinProgResult r = linprog(lp);
//             std::vector<LinProgResult> rs = linprog_costs(lp, {f1, f2});
// Operations: linprog, linprog_costs (linprog.cpp)
// See also:   contDynamics/linear_sys/private/priv_verify_ra_zonotope.cpp
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <limits>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A linear program; an empty bound vector means no bound (-inf / +inf), and an empty
/// constraint matrix means no constraints of that kind.
struct LinProg {
    int n = 0;                        ///< number of variables
    std::vector<double> f;            ///< cost (n)
    std::vector<double> Aineq, bineq; ///< (m_ineq, n) row-major, and (m_ineq)
    std::vector<double> Aeq, beq;     ///< (m_eq, n) row-major, and (m_eq)
    std::vector<double> lb, ub;       ///< (n) each, or empty
};

/// The outcome: x, fval and the multipliers are only set if the status is Optimal. The multipliers
/// of f'x + lambda'(Aineq x - bineq) + mu'(Aeq x - beq) at the optimal basis: lambda >= 0, mu free.
/// Candidates only, to be checked by weak duality; the multipliers of the bounds are not given.
struct LinProgResult {
    enum class Status { optimal, infeasible, unbounded };
    Status status = Status::infeasible;
    std::vector<double> x;
    double fval = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> dual_ineq, dual_eq;
};

/// The minimizer and minimal value of the program, or the reason there is none.
LinProgResult linprog(const LinProg &lp);

/// The programs of lp (its cost f is not used) with each of the costs, one result per cost. The
/// results are those of linprog with that cost; the constraints are prepared (and phase 1 of the
/// simplex method run) once, which the costs share.
std::vector<LinProgResult> linprog_costs(const LinProg &lp,
                                         const std::vector<std::vector<double>> &costs);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
