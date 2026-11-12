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

// The box as a zonotope ---------------------------------------------------------------------------

/// The zonotope of the box [lo, hi]; dense keeps a generator per axis, a flat axis too, as
/// CORALean's Interval.zonotope (CORA's drops the axes without a radius).
Zonotope aux_box_zonotope(const std::vector<double> &lo, const std::vector<double> &hi,
                          const Tensor &like, bool dense) {
    const int64_t n = static_cast<int64_t>(lo.size());
    const Interval box(Tensor::like(like, lo, {n, 1}), Tensor::like(like, hi, {n, 1}));
    if (!dense) return box.zonotope();
    const Tensor center = box.center();
    return Zonotope(center, (box.sup() - center).maximum(center - box.inf()).diag());
}

// The joined cuts ---------------------------------------------------------------------------------

using Box = std::pair<std::vector<double>, std::vector<double>>;

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

// The hull of the sets ----------------------------------------------------------------------------

/// The box of a zonotope, c -/+ the sum of the absolute generators (summed from the last one)
/// (CORALean's Zonotope.interval, without an error box).
Box aux_box_of(const Zonotope &Z) {
    const std::vector<double> c = Z.center().data(), g = Z.generators().data();
    const std::size_t n = c.size(), h = g.size() / (n == 0 ? 1 : n);
    Box box{std::vector<double>(n), std::vector<double>(n)};
    for (std::size_t i = 0; i < n; ++i) {
        double rad = 0.0;
        for (std::size_t j = h; j-- > 0;) rad = std::fabs(g[i * h + j]) + rad;
        box.first[i] = (c[i] - rad) + 0.0;
        box.second[i] = (c[i] + rad) + 0.0;
    }
    return box;
}

/// The boxes of all sets, joined (the members of a bundle intersected): it holds every state of R.
Box aux_hull(const std::vector<Set> &R) {
    Box hull;
    for (const Set &S : R) {
        // the box of one set: its members' boxes, intersected
        Box box;
        for (const Zonotope &Z : priv_zonotopes(S)) {
            const Box member = aux_box_of(Z);
            if (box.first.empty()) box = member;
            for (std::size_t k = 0; k < box.first.size(); ++k) {
                box.first[k] = std::max(box.first[k], member.first[k]);
                box.second[k] = std::min(box.second[k], member.second[k]);
            }
        }
        // joined with the boxes of the sets before
        if (hull.first.empty()) hull = box;
        for (std::size_t k = 0; k < box.first.size(); ++k) {
            hull.first[k] = std::min(hull.first[k], box.first[k]);
            hull.second[k] = std::max(hull.second[k], box.second[k]);
        }
    }
    return hull;
}

// The defect --------------------------------------------------------------------------------------

/// |I - B B'|, entry by entry, or none where B B' = I exactly (CORALean's step_orthDefect, the
/// dot products summed from the last entry).
std::optional<std::vector<double>> aux_defect_magnitude(const Tensor &basis) {
    const std::size_t m = static_cast<std::size_t>(basis.shape()[0]);
    const std::vector<double> B = basis.data();
    std::vector<double> mag(m * m);
    bool zero = true;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (std::size_t k = m; k-- > 0;) {
                const double prod = B[i * m + k] * B[j * m + k];
                dot = prod + dot;
            }
            const double e = (i == j ? 1.0 : 0.0) - dot;
            if (e != 0.0) zero = false;
            mag[i * m + j] = std::fabs(e);
        }
    }
    if (zero) return std::nullopt;
    return mag;
}

/// The radius r of the box [-r, r] of (I - B B') x over the states x of the cuts:
/// r_i = sum_j |I - B B'|_ij |X|_j, X the hull of the sets, summed from the last term (CORALean's
/// Location.zonoGirardMemberFloat).
std::vector<double> aux_defect_radius(const std::vector<double> &mag, const Box &X) {
    const std::size_t m = X.first.size();
    std::vector<double> r(m);
    for (std::size_t i = 0; i < m; ++i) {
        double acc = 0.0;
        for (std::size_t j = m; j-- > 0;) {
            const double mag_x = std::max(std::fabs(X.first[j]), std::fabs(X.second[j]));
            const double term = mag[i * m + j] * mag_x;
            acc = term + acc;
        }
        r[i] = acc;
    }
    return r;
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
    std::optional<Box> coordinate; // the hull of the sets, for the defect
    for (const Tensor &basis : B) {
        const Box box = aux_joined(R, normal, gamma, basis);
        const std::vector<double> &lo = box.first, &hi = box.second;
        // a box with inverted ends is an empty cut
        for (std::size_t k = 0; k < lo.size(); ++k)
            if (lo[k] > hi[k]) return std::nullopt;
        Zonotope Z = aux_box_zonotope(lo, hi, basis, dense).linear_map(basis);
        // a basis that is not orthogonal adds the box of (I - B B') x
        if (const auto mag = aux_defect_magnitude(basis)) {
            if (!coordinate) coordinate = aux_hull(R);
            const std::vector<double> r = aux_defect_radius(*mag, *coordinate);
            std::vector<double> negr(r.size());
            for (std::size_t k = 0; k < r.size(); ++k) negr[k] = -r[k];
            Z = Z.plus(aux_box_zonotope(negr, r, basis, dense));
        }
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
