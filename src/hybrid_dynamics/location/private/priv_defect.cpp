// priv_defect - the box of (I - B B') x, for a basis that is orthogonal only up to rounding
//
// A basis computed in floats (pca) has B B' = I only up to rounding. A state x is B (B' x) plus
// (I - B B') x, so a set that holds B (B' x) is widened by the box [-r, r], r_i = sum_j |I - B
// B'|_ij |X|_j, X the hull of the sets that hold x (CORALean's Location.orthDefectFloat). Where B
// B' is exactly the identity there is no box, and the set stays as it is.
//
// Syntax:   std::optional<Zonotope> D = priv_defect(basis, R, dense);
// Inputs:   basis - the basis (n, n)
//           R - the sets that hold the states
//           dense - a generator per axis, as CORALean's Interval.zonotope
// Outputs:  D - the zonotope of [-r, r]; none where B B' = I exactly
// See also: priv_guard_intersect_zono_girard, priv_guard_intersect_nondet_guard
//
// CORA.cpp only: CORA assumes B B' = I; CORALean adds the defect (Location.orthDefectFloat)
// Authors:       Claude Opus 5.5, Claude Sonnet 5.5
// Written:       09-November-2026
// Last update:   12-November-2026
// Reviewed:      ---

#include "hybrid_dynamics/location/private/priv.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

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
std::optional<std::vector<double>> aux_magnitude(const Tensor &basis) {
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

/// The radius r of the box [-r, r] of (I - B B') x over the states x of the hull X:
/// r_i = sum_j |I - B B'|_ij |X|_j, summed from the last term (CORALean's
/// Location.orthDefectFloat).
std::vector<double> aux_radius(const std::vector<double> &mag, const Box &X) {
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

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope priv_box_zonotope(const Interval &box, bool dense) {
    if (!dense) return box.zonotope();
    const Tensor center = box.center();
    return Zonotope(center, (box.sup() - center).maximum(center - box.inf()).diag());
}

Zonotope priv_box_zonotope(const std::vector<double> &lo, const std::vector<double> &hi,
                           const Tensor &like, bool dense) {
    const int64_t n = static_cast<int64_t>(lo.size());
    return priv_box_zonotope(
        Interval(Tensor::like(like, lo, {n, 1}), Tensor::like(like, hi, {n, 1})), dense);
}

std::optional<Zonotope> priv_defect(const Tensor &basis, const std::vector<Set> &R, bool dense) {
    const std::optional<std::vector<double>> mag = aux_magnitude(basis);
    if (!mag) return std::nullopt;
    const std::vector<double> r = aux_radius(*mag, aux_hull(R));
    std::vector<double> negr(r.size());
    for (std::size_t k = 0; k < r.size(); ++k) negr[k] = -r[k];
    return priv_box_zonotope(negr, r, basis, dense);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
