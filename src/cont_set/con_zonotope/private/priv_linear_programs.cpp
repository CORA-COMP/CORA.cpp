// priv_linear_programs - the linear programs of a constrained zonotope: ranges and emptiness
//
// The programs are those of the factor polytope in the unit box (Polytope::LinearPrograms); here
// only the affine map c + G x is added.
//
// Syntax:   HostConZonotope H = priv_host_con_zonotope(cz);   auto r = priv_range(H, d);
//           bool empty = priv_is_empty(H);   auto lo_hi = priv_factor_ranges(H);
// Inputs:   H - a single constrained zonotope on the host
//           d - a direction (n values)
// Outputs:  R - the range of a linear function, a support point or whether no factor vector exists
// See also: priv_blocks, Polytope::LinearPrograms
//
// CORA.cpp only: MATLAB hands its matrices to CORAlinprog as they are
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The cost G'd of the function d'(c + G x), the generator part.
std::vector<double> aux_cost(const HostConZonotope &H, const std::vector<double> &d) {
    std::vector<double> f(std::size_t(H.m), 0.0);
    for (int64_t j = 0; j < H.m; ++j)
        for (int64_t i = 0; i < H.n; ++i) f[j] += d[i] * H.G[i * H.m + j];
    return f;
}

/// The offset d'c of the function d'(c + G x).
double aux_offset(const HostConZonotope &H, const std::vector<double> &d) {
    double offset = 0.0;
    for (int64_t i = 0; i < H.n; ++i) offset += d[i] * H.c[i];
    return offset;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Host copy ---------------------------------------------------------------------------------------

HostConZonotope priv_host_con_zonotope(const ConZonotope &cz) {
    // the programs run on one set; a batch has no single constraint matrix
    if (cz.c().ndim() != 2)
        CORA_FAIL(Error::not_supported, "ConZonotope: a linear program per set supports a single "
                                        "set, not a batch");
    return {cz.dim(),
            cz.generators().shape().back(),
            cz.A().shape()[0],
            cz.c().data(),
            cz.generators().data(),
            cz.A().data(),
            cz.b().data(),
            cz.factor_programs()};
}

// Ranges and supports -----------------------------------------------------------------------------

std::optional<std::pair<double, double>> priv_range(const HostConZonotope &H,
                                                    const std::vector<double> &d) {
    const std::optional<std::pair<double, double>> r = H.factors.range(aux_cost(H, d));
    if (!r) return std::nullopt;
    const double offset = aux_offset(H, d);
    return std::pair{offset + r->first, offset + r->second};
}

std::optional<PrivSupport> priv_support(const HostConZonotope &H, const std::vector<double> &d) {
    const std::optional<Polytope::LinearPrograms::Maximum> best = H.factors.maximum(aux_cost(H, d));
    if (!best) return std::nullopt;
    // the point of the factors the program found: c + G x
    std::vector<double> point = H.c;
    for (int64_t i = 0; i < H.n; ++i)
        for (int64_t j = 0; j < H.m; ++j) point[i] += H.G[i * H.m + j] * best->x[j];
    return PrivSupport{aux_offset(H, d) + best->value, point};
}

// Emptiness and ranges ----------------------------------------------------------------------------

bool priv_is_empty(const HostConZonotope &H) { return H.factors.is_empty(); }

std::pair<std::vector<double>, std::vector<double>> priv_factor_ranges(const HostConZonotope &H) {
    return H.factors.bounds();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
