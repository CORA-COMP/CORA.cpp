// priv_zonotope_norm - the norm of a point w.r.t. the generators of a zonotope
//
// The smallest s with p = G beta and |beta|_inf <= s: the factor by which the zonotope centered
// at the origin must be scaled to reach p. A linear program in (s, beta).
//
// Syntax:   double s = priv_zonotope_norm(Z, p);
//           ZonotopeNorm r = priv_zonotope_norm_minimizer(Z, p);
// Inputs:   Z - zonotope (its center is ignored)
//           p - a point (n values)
// Outputs:  s - the norm; infinity if p is not in the span of the generators
//           r - s and the coefficients beta (none if s is infinite)
// See also: priv_containment_vertex_enumeration, zonotope_norm
//
// CORA.cpp only: the linear program of zonotopeNorm, on the host and unchecked, for contains and
//                the other callers
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/global/linprog.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonotopeNorm priv_zonotope_norm_minimizer(const HostZonotope &Z, const std::vector<double> &p) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    if (Z.m == 0) {
        for (const double x : p)
            if (x != 0) return {kInf, {}};
        return {0.0, {}};
    }
    // variables (s, beta): min s, G beta = p, -s <= beta_i <= s
    LinProg lp;
    lp.n = int(1 + Z.m);
    lp.f.assign(lp.n, 0.0);
    lp.f[0] = 1.0;
    lp.Aeq.assign(std::size_t(Z.n * lp.n), 0.0);
    for (int64_t i = 0; i < Z.n; ++i)
        for (int64_t j = 0; j < Z.m; ++j) lp.Aeq[i * lp.n + 1 + j] = Z.G[i * Z.m + j];
    lp.beq = p;
    lp.Aineq.assign(std::size_t(2 * Z.m * lp.n), 0.0);
    lp.bineq.assign(std::size_t(2 * Z.m), 0.0);
    for (int64_t j = 0; j < Z.m; ++j) {
        lp.Aineq[(2 * j) * lp.n] = lp.Aineq[(2 * j + 1) * lp.n] = -1.0;
        lp.Aineq[(2 * j) * lp.n + 1 + j] = 1.0;
        lp.Aineq[(2 * j + 1) * lp.n + 1 + j] = -1.0;
    }
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal) return {kInf, {}};
    return {r.fval, std::vector<double>(r.x.begin() + 1, r.x.end())};
}

double priv_zonotope_norm(const HostZonotope &Z, const std::vector<double> &p) {
    // the value of the minimizer
    return priv_zonotope_norm_minimizer(Z, p).value;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
