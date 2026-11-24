// robustness - how robustly points satisfy a specification, as CORA's robustness
//
// The signed distance of a point to the boundary of the specification's set: positive where the
// point satisfies a safe set or invariant (the distance to the nearest facet hyperplane, rows
// normalized) or violates nothing, negative where it does not (the distance to the set: to the one
// violated hyperplane, to the vertex of n violated hyperplanes, else the 1-norm distance by a
// linear program); an unsafe set reverses the signs. Equalities of the set are two inequalities
// (CORA ignores them: a point off the set of a lower dimension gets 0). A custom specification is
// Error::not_supported. Values of a trajectory count only inside the window of the specification.
//
// Syntax:   values = robustness(spec, points);   val = robustness(spec, points, times);
// Inputs:   spec - a specification (or a list: the least value)
//           points - (N, n), one point per row
//           times - one time per point
// Outputs:  values - one value per point
//           val - the least value over the points in the window
// See also: Specification::check_points
//
// CORA: specification/@specification/robustness.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/linprog.h"
#include "cora/global/within_tol.h"
#include "cora/specification/specification.h"
#include "specification/private/priv_polytope_of.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The halfspaces and the distance to them ---------------------------------------------------------

/// The halfspaces C x <= d of the set with unit rows; an equality is two of them.
struct Halfspaces {
    int64_t n = 0;
    std::vector<double> C, d; ///< (m, n) row-major, (m)
};

Halfspaces aux_halfspaces(const Polytope &P) {
    const Polytope::Constraints c = P.constraints();
    std::vector<Tensor> A{c.A}, b{c.b};
    if (c.Ae.numel() > 0) {
        A.push_back(c.Ae);
        b.push_back(c.be);
        A.push_back(-c.Ae);
        b.push_back(-c.be);
    }
    const Polytope unit =
        Polytope(Tensor::concat(A, 0), Tensor::concat(b, 0)).normalize_constraints("A");
    if (unit.b().numel() == 0)
        CORA_FAIL(Error::wrong_value, "robustness: the set of the specification has no halfspaces");
    return {unit.A().shape()[1], unit.A().data(), unit.b().data()};
}

/// Solves the square system M v = r by Gaussian elimination with pivoting; false if singular.
bool aux_solve(std::vector<double> M, std::vector<double> r, int64_t n, std::vector<double> &v) {
    const std::size_t N = static_cast<std::size_t>(n);
    double scale = 0;
    for (const double e : M) scale = std::max(scale, std::abs(e));
    // eliminate column c below the pivot of largest magnitude
    for (std::size_t c = 0; c < N; ++c) {
        std::size_t piv = c;
        for (std::size_t i = c + 1; i < N; ++i)
            if (std::abs(M[i * N + c]) > std::abs(M[piv * N + c])) piv = i;
        if (std::abs(M[piv * N + c]) <= machine_eps() * scale) return false;
        for (std::size_t j = 0; j < N; ++j) std::swap(M[c * N + j], M[piv * N + j]);
        std::swap(r[c], r[piv]);
        for (std::size_t i = c + 1; i < N; ++i) {
            const double f = M[i * N + c] / M[c * N + c];
            for (std::size_t j = c; j < N; ++j) M[i * N + j] -= f * M[c * N + j];
            r[i] -= f * r[c];
        }
    }
    // back substitution
    v.assign(N, 0.0);
    for (std::size_t i = N; i-- > 0;) {
        double sum = r[i];
        for (std::size_t j = i + 1; j < N; ++j) sum -= M[i * N + j] * v[j];
        v[i] = sum / M[i * N + i];
    }
    return true;
}

/// The 1-norm distance of p to {x | C x <= d}: min sum(s + t) with x + s - t = p, s, t >= 0.
double aux_distance_one_norm(const Halfspaces &H, const std::vector<double> &p) {
    const std::size_t n = static_cast<std::size_t>(H.n), m = H.d.size();
    LinProg lp;
    lp.n = static_cast<int>(3 * n);
    lp.f.assign(3 * n, 1.0);
    std::fill(lp.f.begin(), lp.f.begin() + n, 0.0);
    lp.Aeq.assign(n * 3 * n, 0.0);
    lp.beq = p;
    for (std::size_t i = 0; i < n; ++i) {
        lp.Aeq[i * 3 * n + i] = 1.0;
        lp.Aeq[i * 3 * n + n + i] = 1.0;
        lp.Aeq[i * 3 * n + 2 * n + i] = -1.0;
    }
    // C x <= d, and s, t >= 0 as -s <= 0, -t <= 0
    lp.Aineq.assign((m + 2 * n) * 3 * n, 0.0);
    lp.bineq.assign(m + 2 * n, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) lp.Aineq[i * 3 * n + j] = H.C[i * n + j];
        lp.bineq[i] = H.d[i];
    }
    for (std::size_t i = 0; i < 2 * n; ++i) lp.Aineq[(m + i) * 3 * n + n + i] = -1.0;
    const LinProgResult r = linprog(lp);
    if (r.status != LinProgResult::Status::optimal)
        CORA_FAIL(Error::solver_issue, "robustness: the distance to the set could not be computed");
    return r.fval;
}

/// The distance of the point p outside {x | C x <= d} (CORA's aux_distancePolyPoint).
double aux_distance(const Halfspaces &H, const std::vector<double> &p) {
    const std::size_t n = static_cast<std::size_t>(H.n), m = H.d.size();
    std::vector<std::size_t> violated;
    std::vector<double> excess(m);
    for (std::size_t i = 0; i < m; ++i) {
        double v = -H.d[i];
        for (std::size_t j = 0; j < n; ++j) v += H.C[i * n + j] * p[j];
        excess[i] = v;
        if (v > 0) violated.push_back(i);
    }
    if (violated.size() == 1) return excess[violated[0]]; // the distance to that hyperplane
    if (violated.size() == n) {                           // the vertex of the violated hyperplanes
        std::vector<double> M, r, v;
        for (const std::size_t i : violated) {
            M.insert(M.end(), H.C.begin() + i * n, H.C.begin() + (i + 1) * n);
            r.push_back(H.d[i]);
        }
        if (aux_solve(M, r, H.n, v)) {
            double sum = 0;
            for (std::size_t j = 0; j < n; ++j) sum += (v[j] - p[j]) * (v[j] - p[j]);
            return std::sqrt(sum);
        }
    }
    return aux_distance_one_norm(H, p);
}

// The points and their window ---------------------------------------------------------------------

/// The robustness of one point.
double aux_point(const Specification &spec, const Polytope &P, const Halfspaces &H,
                 const Tensor &p) {
    const std::vector<double> x = p.data();
    const bool inside = P.contains(p);
    double nearest = 0;
    if (inside) {
        nearest = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < H.d.size(); ++i) {
            double v = -H.d[i];
            for (std::size_t j = 0; j < x.size(); ++j) v += H.C[i * x.size() + j] * x[j];
            nearest = std::min(nearest, std::abs(v));
        }
    } else {
        nearest = aux_distance(H, x);
    }
    const bool safe = spec.type() != SpecType::unsafe_set;
    // inside a safe set is positive, outside it negative; an unsafe set the other way around
    return inside == safe ? nearest : -nearest;
}

/// The values of the points in the window, with their times (those of the interpolated one).
void aux_window(const Specification &spec, const Tensor &points, const std::vector<double> &times,
                double tol, std::vector<std::size_t> &active, Tensor &used) {
    const std::size_t N = static_cast<std::size_t>(points.shape()[0]);
    const int64_t n = points.shape()[1];
    used = points;
    for (std::size_t i = 0; i < N; ++i)
        if (!spec.time() ||
            (spec.time()->first - tol <= times[i] && times[i] <= spec.time()->second + tol))
            active.push_back(i);
    if (!spec.time() || !active.empty()) return;
    // no point in the window: linear interpolation at its center
    const double center = 0.5 * (spec.time()->first + spec.time()->second);
    std::vector<std::size_t> order(N);
    for (std::size_t i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return times[a] < times[b]; });
    if (N < 2 || center < times[order.front()] || center > times[order.back()])
        CORA_FAIL(Error::special_error, "robustness: the window lies outside the times given");
    std::size_t hi = 1;
    while (hi + 1 < N && times[order[hi]] < center) ++hi;
    const std::size_t a = order[hi - 1], b = order[hi];
    const double w = (center - times[a]) / (times[b] - times[a]);
    const std::vector<double> d = points.data();
    std::vector<double> row(static_cast<std::size_t>(n));
    for (std::size_t j = 0; j < row.size(); ++j)
        row[j] = (1 - w) * d[a * row.size() + j] + w * d[b * row.size() + j];
    used = Tensor::like(points, row, {1, n});
    active = {0};
}

/// The values of the points the specification is active for.
std::vector<double> aux_values(const Specification &spec, const Tensor &points,
                               const std::vector<double> &times, double tol) {
    // the inputs, which the public entries pass on
    if (spec.type() == SpecType::custom)
        CORA_FAIL(Error::not_supported, "robustness: not supported for custom specifications");
    if (points.ndim() != 2)
        CORA_FAIL(Error::wrong_value, "robustness: give the points as (N, n), one per row");
    if (spec.time() && times.size() != static_cast<std::size_t>(points.shape()[0]))
        CORA_FAIL(Error::special_error,
                  "robustness: a timed specification needs one time per point");
    const Polytope P = priv_polytope_of(spec.set());
    const Halfspaces H = aux_halfspaces(P);
    std::vector<std::size_t> active;
    Tensor used;
    aux_window(spec, points, times, tol, active, used);
    const int64_t n = used.shape()[1];
    const std::vector<double> d = used.data();
    // the value of each point the specification is active for
    std::vector<double> out;
    for (const std::size_t i : active) {
        const std::vector<double> row(d.begin() + i * n, d.begin() + (i + 1) * n);
        out.push_back(aux_point(spec, P, H, Tensor::like(points, row, {n})));
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Of one specification and of a list --------------------------------------------------------------

Tensor robustness(const Specification &spec, const Tensor &points) {
    detail::Entry entry;
    const double tol = resolve_tol(entry, std::nullopt, "robustness");
    const std::vector<double> v = aux_values(spec, points, {}, tol);
    return Tensor::like(points, v, {static_cast<int64_t>(v.size())});
}

double robustness(const Specification &spec, const Tensor &points,
                  const std::vector<double> &times) {
    detail::Entry entry;
    const double tol = resolve_tol(entry, std::nullopt, "robustness");
    CORA_CHECK(times.size() == static_cast<std::size_t>(points.ndim() == 2 ? points.shape()[0] : 0),
               Error::special_error,
               [] { return std::string("robustness: give one time per point"); });
    // the least value over the points in the window
    double val = std::numeric_limits<double>::infinity();
    for (const double v : aux_values(spec, points, times, tol)) val = std::min(val, v);
    // the least value is the robustness of the trajectory
    return val;
}

Tensor robustness(const std::vector<Specification> &specs, const Tensor &points) {
    detail::Entry entry;
    const double tol = resolve_tol(entry, std::nullopt, "robustness");
    // the least value over the specifications, point by point
    std::vector<double> val;
    for (std::size_t k = 0; k < specs.size(); ++k) {
        const std::vector<double> v = aux_values(specs[k], points, {}, tol);
        if (k == 0) val = v;
        CORA_CHECK(v.size() == val.size(), Error::special_error, [] {
            return std::string("robustness: the specifications are active for different points");
        });
        for (std::size_t i = 0; i < v.size(); ++i) val[i] = std::min(val[i], v[i]);
    }
    return Tensor::like(points, val, {static_cast<int64_t>(val.size())});
}

double robustness(const std::vector<Specification> &specs, const Tensor &points,
                  const std::vector<double> &times) {
    detail::Entry entry;
    const double tol = resolve_tol(entry, std::nullopt, "robustness");
    double val = std::numeric_limits<double>::infinity();
    for (const Specification &spec : specs)
        for (const double v : aux_values(spec, points, times, tol)) val = std::min(val, v);
    return val;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
