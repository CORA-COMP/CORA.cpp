// priv_linear_programs - the linear programs of a constrained zonotope: ranges, emptiness, center
//
// Syntax:   HostConZonotope H = priv_host_con_zonotope(cz);   auto r = priv_range(H, d);
//           bool empty = priv_is_empty(H);   auto lo_hi = priv_factor_ranges(H);
//           auto x = priv_chebyshev_center(Aineq, bineq, Aeq, beq, k);
// Inputs:   H - a single constrained zonotope on the host;  d - a direction (n values)
// Outputs:  the range of a linear function, whether no factor vector exists, the range of every
//           factor, the Chebyshev center of a polytope
// See also: priv_blocks, linprog
//
// CORA.cpp only: MATLAB hands its matrices to CORAlinprog as they are

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/linprog.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Programs ----------------------------------------------------------------------------------------

/// The program min f'x over |x|_inf <= 1, A x = b with the cost f (m values).
LinProgResult aux_solve(const HostConZonotope &H, const std::vector<double> &f) {
    // without factors there is nothing to optimize: the constraints read 0 = b
    if (H.m == 0) {
        LinProgResult none;
        const bool feasible = std::all_of(H.b.begin(), H.b.end(), [](double x) {
            return std::abs(x) <= numerics::kSimplexFeasible;
        });
        none.status = feasible ? LinProgResult::Status::optimal : LinProgResult::Status::infeasible;
        none.fval = feasible ? 0.0 : none.fval;
        return none;
    }
    LinProg lp;
    lp.n = int(H.m);
    lp.f = f;
    lp.Aeq = H.A;
    lp.beq = H.b;
    lp.lb.assign(std::size_t(H.m), -1.0);
    lp.ub.assign(std::size_t(H.m), 1.0);
    return linprog(lp);
}

/// The cost G'd of the function d'(c + G x), the generator part.
std::vector<double> aux_cost(const HostConZonotope &H, const std::vector<double> &d) {
    std::vector<double> f(std::size_t(H.m), 0.0);
    for (int64_t j = 0; j < H.m; ++j)
        for (int64_t i = 0; i < H.n; ++i) f[j] += d[i] * H.G[i * H.m + j];
    return f;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Host copy ---------------------------------------------------------------------------------------

HostConZonotope priv_host_con_zonotope(const ConZonotope &cz) {
    if (cz.c().ndim() != 2)
        CORA_FAIL(Error::not_supported, "ConZonotope: a linear program per set supports a single "
                                        "set, not a batch");
    return {cz.dim(),      cz.generators().shape().back(), cz.A().shape()[0],
            cz.c().data(), cz.generators().data(),         cz.A().data(),
            cz.b().data()};
}

std::optional<std::pair<double, double>> priv_range(const HostConZonotope &H,
                                                    const std::vector<double> &d) {
    double offset = 0.0;
    for (int64_t i = 0; i < H.n; ++i) offset += d[i] * H.c[i];
    std::vector<double> f = aux_cost(H, d);
    const LinProgResult lower = aux_solve(H, f);
    if (lower.status != LinProgResult::Status::optimal) return std::nullopt;
    for (double &x : f) x = -x;
    const LinProgResult upper = aux_solve(H, f);
    if (upper.status != LinProgResult::Status::optimal) return std::nullopt;
    return std::pair{offset + lower.fval, offset - upper.fval};
}

// Emptiness and ranges ----------------------------------------------------------------------------

bool priv_is_empty(const HostConZonotope &H) {
    if (H.p == 0) return false;
    // the empty set has no point in the box that meets the constraints
    return aux_solve(H, std::vector<double>(std::size_t(H.m), 0.0)).status !=
           LinProgResult::Status::optimal;
}

std::pair<std::vector<double>, std::vector<double>> priv_factor_ranges(const HostConZonotope &H) {
    std::vector<double> lo(std::size_t(H.m)), hi(std::size_t(H.m));
    for (int64_t j = 0; j < H.m; ++j) {
        std::vector<double> f(std::size_t(H.m), 0.0);
        f[j] = 1.0;
        lo[j] = aux_solve(H, f).fval;
        f[j] = -1.0;
        hi[j] = -aux_solve(H, f).fval;
    }
    return {lo, hi};
}

std::optional<std::vector<double>>
priv_chebyshev_center(const std::vector<double> &Aineq, const std::vector<double> &bineq,
                      const std::vector<double> &Aeq, const std::vector<double> &beq, int64_t k) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    // variables (x, r): maximize the radius r of the ball, a_i x + r |a_i| <= b_i, A x = b, r >= 0
    LinProg lp;
    lp.n = int(k + 1);
    lp.f.assign(std::size_t(lp.n), 0.0);
    lp.f[k] = -1.0;
    const int64_t rows = int64_t(bineq.size()), rows_eq = int64_t(beq.size());
    lp.Aineq.assign(std::size_t(rows * lp.n), 0.0);
    lp.bineq = bineq;
    for (int64_t i = 0; i < rows; ++i) {
        double norm = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            lp.Aineq[i * lp.n + j] = Aineq[i * k + j];
            norm += Aineq[i * k + j] * Aineq[i * k + j];
        }
        lp.Aineq[i * lp.n + k] = std::sqrt(norm);
    }
    lp.Aeq.assign(std::size_t(rows_eq * lp.n), 0.0);
    lp.beq = beq;
    for (int64_t i = 0; i < rows_eq; ++i)
        for (int64_t j = 0; j < k; ++j) lp.Aeq[i * lp.n + j] = Aeq[i * k + j];
    lp.lb.assign(std::size_t(lp.n), -kInf);
    lp.lb[k] = 0.0;
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal) return std::nullopt;
    return std::vector<double>(r.x.begin(), r.x.begin() + k);
}
// the program of the largest ball inside the constraints

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
