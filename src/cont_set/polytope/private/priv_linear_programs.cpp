// priv_linear_programs - the linear programs of a polytope: support values, emptiness, center
//
// The Chebyshev center, boundedness, membership in a hull and the extreme points of a cloud are
// programs too.
//
// Syntax:   Support s = priv_support(H, direction, true);   bool empty = priv_is_empty(H);
//           bool bounded = priv_is_bounded(H);   auto c = priv_chebyshev_center(H);
//           bool in = priv_in_hull(points, x);   auto kept = priv_extreme_points(points);
// Inputs:   H - a single polytope in halfspace form
//           points - rows of n numbers
// Outputs:  the optimum of a linear function, whether the set has a point, whether it is bounded,
//           the center of its largest ball, whether a point is a convex combination, the indices
//           of the extreme points
// See also: priv_compact, priv_vertices, linprog
//
// CORA: contSet/@polytope/private/priv_supportFunc.m
// CORA: contSet/@polytope/private/priv_representsa_emptySet.m
// CORA: contSet/@polytope/private/priv_feasiblePoint.m
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/linprog.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The rows row-major.
std::vector<double> aux_flatten(const Rows &rows) {
    std::vector<double> flat;
    for (const Row &row : rows) flat.insert(flat.end(), row.begin(), row.end());
    return flat;
}

/// The program min f'x over the polytope H, free variables.
LinProgResult aux_solve(const HostH &H, const Row &f) {
    LinProg lp;
    lp.n = int(H.n);
    lp.f = f;
    lp.Aineq = aux_flatten(H.A);
    lp.bineq = H.b;
    lp.Aeq = aux_flatten(H.Ae);
    lp.beq = H.be;
    return linprog(lp);
}

/// Whether a row 0 <= b or 0 = b cannot hold (the row has only zeros).
bool aux_infeasible_zero_row(const HostH &H) {
    const auto zero_row = [](const Row &a) {
        return std::all_of(a.begin(), a.end(),
                           [](double x) { return std::abs(x) <= machine_eps(); });
    };
    for (std::size_t i = 0; i < H.A.size(); ++i)
        if (zero_row(H.A[i]) && H.b[i] < 0.0) return true;
    for (std::size_t i = 0; i < H.Ae.size(); ++i)
        if (zero_row(H.Ae[i]) && std::abs(H.be[i]) > machine_eps()) return true;
    return false;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Support values ----------------------------------------------------------------------------------

Support priv_support(const HostH &H, const Row &direction, bool upper) {
    Support s;
    // without any constraint the set is R^n
    if (H.A.empty() && H.Ae.empty()) {
        s.status = Support::Status::unbounded;
        s.value = upper ? kInf : -kInf;
        return s;
    }
    Row f = direction;
    if (upper)
        for (double &x : f) x = -x;
    const LinProgResult r = aux_solve(H, f);
    if (r.status == LinProgResult::Status::optimal) {
        s.status = Support::Status::optimal;
        s.value = upper ? -r.fval : r.fval;
        s.x = r.x;
    } else if (r.status == LinProgResult::Status::infeasible) {
        s.status = Support::Status::infeasible;
        s.value = upper ? -kInf : kInf;
    } else {
        s.status = Support::Status::unbounded;
        s.value = upper ? kInf : -kInf;
        // the point at infinity along the improving direction
        s.x.resize(f.size());
        for (std::size_t i = 0; i < f.size(); ++i)
            s.x[i] = f[i] > 0 ? -kInf : (f[i] < 0 ? kInf : 0.0);
    }
    return s;
}

// Emptiness and boundedness -----------------------------------------------------------------------

bool priv_is_empty(const HostH &H) {
    if (H.n == 1) {
        bool empty = false;
        priv_vertices_1d(H, empty);
        return empty;
    }
    if (H.b.empty() && H.be.empty()) return false;
    if (aux_infeasible_zero_row(H)) return true;
    if (H.n == 0) return false;
    return aux_solve(H, Row(std::size_t(H.n), 0.0)).status == LinProgResult::Status::infeasible;
}

bool priv_is_bounded(const HostH &H) {
    if (H.n == 1) {
        bool empty = false;
        const Row v = priv_vertices_1d(H, empty);
        return std::none_of(v.begin(), v.end(), [](double x) { return std::isinf(x); });
    }
    // a coordinate that no row restricts is free
    for (int64_t j = 0; j < H.n; ++j) {
        bool restricted = false;
        for (const Row &a : H.A) restricted = restricted || a[j] != 0.0;
        for (const Row &a : H.Ae) restricted = restricted || a[j] != 0.0;
        if (!restricted) return false;
    }
    // the support function along the axes and the diagonal of a simplex is finite
    for (int64_t i = 0; i <= H.n; ++i) {
        Row direction(std::size_t(H.n), i < H.n ? 0.0 : -1.0);
        if (i < H.n) direction[i] = 1.0;
        const Support s = priv_support(H, direction, true);
        if (s.status == Support::Status::unbounded) return false;
        if (s.status == Support::Status::infeasible) return true;
    }
    return true;
}

std::optional<Row> priv_chebyshev_center(const HostH &H) {
    // variables (x, r): maximize the radius r, a_i x + |a_i| r <= b_i, Ae x = be, r >= 0
    LinProg lp;
    lp.n = int(H.n + 1);
    lp.f.assign(std::size_t(lp.n), 0.0);
    lp.f[std::size_t(H.n)] = -1.0;
    for (Row row : H.A) {
        // the row's length multiplies the radius
        double norm = 0.0;
        for (const double a : row) norm += a * a;
        row.push_back(std::sqrt(norm));
        lp.Aineq.insert(lp.Aineq.end(), row.begin(), row.end());
    }
    lp.bineq = H.b;
    // the equalities do not involve the radius
    for (const Row &a : H.Ae) {
        lp.Aeq.insert(lp.Aeq.end(), a.begin(), a.end());
        lp.Aeq.push_back(0.0);
    }
    lp.beq = H.be;
    lp.lb.assign(std::size_t(lp.n), -kInf);
    lp.lb[std::size_t(H.n)] = 0.0;
    const LinProgResult r = linprog(lp);
    // no solution: empty; a radius without bound: the center is not defined
    if (r.status == LinProgResult::Status::infeasible) return std::nullopt;
    if (r.status == LinProgResult::Status::unbounded)
        return Row(std::size_t(H.n), std::numeric_limits<double>::quiet_NaN());
    return Row(r.x.begin(), r.x.begin() + H.n);
}

std::optional<std::pair<Row, Row>> priv_box_bounds(const HostH &H) {
    // the bounds of every coordinate: its largest value first, which reveals an empty set
    Row lower(static_cast<std::size_t>(H.n)), upper(static_cast<std::size_t>(H.n));
    for (int64_t i = 0; i < H.n; ++i) {
        Row direction(std::size_t(H.n), 0.0);
        direction[std::size_t(i)] = 1.0;
        const Support up = priv_support(H, direction, true);
        // no point: the set is empty
        if (up.status == Support::Status::infeasible) return std::nullopt;
        upper[std::size_t(i)] = up.value;
        lower[std::size_t(i)] = priv_support(H, direction, false).value;
    }
    return std::pair{lower, upper};
}

// Hulls of points ---------------------------------------------------------------------------------

bool priv_in_hull(const Rows &points, const Row &x) {
    if (points.empty()) return false;
    const std::size_t n = x.size(), p = points.size();
    // sum_k lambda_k v_k = x, sum_k lambda_k = 1, lambda >= 0
    LinProg lp;
    lp.n = int(p);
    lp.f.assign(p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < p; ++k) lp.Aeq.push_back(points[k][i]);
        lp.beq.push_back(x[i]);
    }
    for (std::size_t k = 0; k < p; ++k) lp.Aeq.push_back(1.0);
    lp.beq.push_back(1.0);
    lp.lb.assign(p, 0.0);
    return linprog(lp).status == LinProgResult::Status::optimal;
}

std::vector<int64_t> priv_extreme_points(const Rows &points) {
    const std::size_t count = points.size();
    std::vector<bool> keep(count, true);
    // from the back, so that of equal points the first stays
    for (std::size_t i = count; i-- > 0;) {
        Rows others;
        for (std::size_t k = 0; k < count; ++k)
            if (k != i && keep[k]) others.push_back(points[k]);
        if (!others.empty() && priv_in_hull(others, points[i])) keep[i] = false;
    }
    std::vector<int64_t> kept;
    for (std::size_t i = 0; i < count; ++i)
        if (keep[i]) kept.push_back(int64_t(i));
    return kept;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
