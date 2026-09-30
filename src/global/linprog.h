// linprog - a dense linear program solver (two-phase simplex), as CORA's CORAlinprog
//
// Solves  min f'x  s.t.  Aineq x <= bineq,  Aeq x = beq,  lb <= x <= ub  for the small programs
// of set computations (a few hundred variables); the matrices are row-major on the host.
//
// Syntax:     LinProgResult r = linprog(lp);
// Operations: linprog (linprog.cpp)
// See also:   contDynamics/linearSys/private/priv_verifyRA_zonotope.cpp

#pragma once

#include <limits>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A linear program; an empty bound vector means no bound (-inf / +inf), and an empty
/// constraint matrix means no constraints of that kind.
struct LinProg {
    int n = 0;                       ///< number of variables
    std::vector<double> f;           ///< cost (n)
    std::vector<double> Aineq, bineq;  ///< (mIneq, n) row-major, and (mIneq)
    std::vector<double> Aeq, beq;    ///< (mEq, n) row-major, and (mEq)
    std::vector<double> lb, ub;      ///< (n) each, or empty
};

/// The outcome: x and fval are only set if the status is Optimal.
struct LinProgResult {
    enum class Status { Optimal, Infeasible, Unbounded };
    Status status = Status::Infeasible;
    std::vector<double> x;
    double fval = std::numeric_limits<double>::quiet_NaN();
};

/// The minimizer and minimal value of the program, or the reason there is none.
LinProgResult linprog(const LinProg &lp);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
