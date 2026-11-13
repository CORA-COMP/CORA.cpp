// priv_guard_intersect_zono_girard - the guard intersection of zonoGirard, as CORALean defines it
//
// For each basis vector l, each set is mapped to the zonogon S = [n'; l'] Z (the plane image of
// Girard and Le Guernic, HSCC 2008), and the upper end of the cut of S by the line x = gamma is
// the least dual bound over the slopes of its generators: for a multiplier mu, h_S(-mu, 1) + mu
// gamma, the support function rounded up (Location.zonoGirardUpperFloat). The cuts of all sets are
// united per basis as the box [-U(-l), U(l)], and mapped back by the basis. The guard's
// inequalities then cut the box (a constrained zonotope, rescaled), which CORALean does not define
// yet. A box with an end below its other end is an empty cut: CORALean assumes none, and the part
// is dropped.
//
// Syntax:   std::optional<Set> S = priv_guard_intersect_zono_girard(R, guard, B);
// Inputs:   R - the sets that cross the guard
//           guard - a hyperplane n'x = gamma with inequalities
//           B - the bases (n, n)
//           dense - a generator per axis in every box, as CORALean's Interval.zonotope
// Outputs:  S - the enclosure (a zonotope, or a bundle for several bases); none if empty
// See also: priv_guard_intersect, priv_calc_basis
//
// CORA: hybridDynamics/@location/guardIntersect_zonoGirard.m
// CORA authors: Stefan Liu, Niklas Kochdumper
// Authors:       Claude Opus 5.5, Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/global/tensor/tolerance.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The dual bound ----------------------------------------------------------------------------------

/// The upper end of l'x over the states x of Z with n'x = gamma (zonoGirardUpperFloat).
double aux_upper(const Zonotope &Z, const Tensor &normal, double gamma, const Tensor &l) {
    // the plane image [n'; l'] Z, a zonogon
    const Zonotope S =
        Z.linear_map(Tensor::concat({normal.matrix_transpose(), l.matrix_transpose()}, -2));
    // the dual bound at the multiplier mu
    const auto dual = [&](double mu) {
        const Tensor direction = Tensor::like(S.center(), {-mu, 1.0}, {2, 1});
        return S.support_function(direction).item() + mu * gamma;
    };
    // the slopes of the generators are the breakpoints of the dual
    const std::vector<double> G = S.generators().data();
    const std::size_t h = G.size() / 2;
    double least = dual(0.0);
    for (std::size_t j = 0; j < h; ++j) {
        const double a = G[h + j], b = G[j];
        const double slope = 0.0 <= b ? (b <= 0.0 ? 0.0 : a / b) : (-a) / (-b);
        least = std::min(least, dual(slope));
    }
    return least;
}

/// The box [-U(-l), U(l)] of the cut of Z for each column l of the basis.
std::pair<std::vector<double>, std::vector<double>> aux_cut(const Zonotope &Z, const Tensor &normal,
                                                            double gamma, const Tensor &basis) {
    const int64_t n = basis.shape()[0];
    std::vector<double> lo(static_cast<std::size_t>(n)), hi(static_cast<std::size_t>(n));
    for (int64_t k = 0; k < n; ++k) {
        const Tensor l = basis.slice(-1, k, k + 1);
        lo[std::size_t(k)] = -aux_upper(Z, normal, gamma, -l);
        hi[std::size_t(k)] = aux_upper(Z, normal, gamma, l);
    }
    return {lo, hi};
}

// The joined cuts ---------------------------------------------------------------------------------

/// The cuts of all sets along the columns of the basis, joined: the least lower and the greatest
/// upper end. A bundle is the intersection of its members: its cut lies in each member's.
Box aux_joined(const std::vector<Set> &R, const Tensor &normal, double gamma, const Tensor &basis) {
    std::vector<double> lo, hi;
    for (const Set &S : R) {
        std::vector<double> l, u;
        // the members' cuts, intersected
        for (const Zonotope &Z : priv_zonotopes(S)) {
            auto [lz, uz] = aux_cut(Z, normal, gamma, basis);
            if (l.empty()) {
                l = lz;
                u = uz;
            }
            for (std::size_t k = 0; k < l.size(); ++k) {
                l[k] = std::max(l[k], lz[k]);
                u[k] = std::min(u[k], uz[k]);
            }
        }
        // joined with the cuts of the sets before
        if (lo.empty()) {
            lo = l;
            hi = u;
        }
        for (std::size_t k = 0; k < l.size(); ++k) {
            lo[k] = std::min(lo[k], l[k]);
            hi[k] = std::max(hi[k], u[k]);
        }
    }
    return {lo, hi};
}

/// CORA's aux_tightenSet: the zonotope cut by the inequalities P of the guard; none if it misses.
std::optional<Zonotope> aux_tighten(const Zonotope &Z, const Polytope &P) {
    if (!P.is_intersecting(Z, "approx", numerics::kGuardMeet)) return std::nullopt;
    const ConZonotope cz = ConZonotope(Z).intersect(P);
    ConZonotope tight = cz;
    try {
        tight = cz.rescale("iter");
    } catch (const Error &) { // CORA falls back to the linear programs
        tight = cz.rescale("exact");
    }
    return Zonotope(tight.c(), tight.generators());
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::optional<Set> priv_guard_intersect_zono_girard(const std::vector<Set> &R,
                                                    const Polytope &guard,
                                                    const std::vector<Tensor> &B, bool dense) {
    const Tensor normal = guard.Ae().slice(-2, 0, 1).matrix_transpose();
    const double gamma = guard.be().data()[0];
    const bool inequalities = guard.A().shape()[0] > 0;
    std::vector<Zonotope> parts;
    // The boxes per basis -------------------------------------------------------------------------
    for (const Tensor &basis : B) {
        const Box box = aux_joined(R, normal, gamma, basis);
        const std::vector<double> &lo = box.first, &hi = box.second;
        // a box with inverted ends is an empty cut
        for (std::size_t k = 0; k < lo.size(); ++k)
            if (lo[k] > hi[k]) return std::nullopt;
        Zonotope Z = priv_box_zonotope(lo, hi, basis, dense).linear_map(basis);
        // a basis that is not orthogonal adds the box of (I - B B') x
        if (const auto defect = priv_defect(basis, R, dense)) Z = Z.plus(*defect);
        if (inequalities) {
            const std::optional<Zonotope> cut = aux_tighten(Z, Polytope(guard.A(), guard.b()));
            if (!cut) continue; // CORA drops the basis
            Z = *cut;
        }
        parts.push_back(std::move(Z));
    }
    if (parts.empty()) return std::nullopt;
    return priv_bundle(std::move(parts));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
