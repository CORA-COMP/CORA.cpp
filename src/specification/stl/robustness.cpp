// robustness - how robustly a trace satisfies a formula
//
// The formula is brought to negation normal form and its non-temporal parts are numbered; each
// part is a union of safe sets (polytopes of linear comparisons, level sets of the others), which
// is turned into a union of unsafe sets. A sample's robustness for a part is the signed distance to
// the unsafe sets, and the temporal operators take the minimum or maximum over the windows, as
// Definition 2.3 of Sankaranarayanan et al. A trace not uniformly sampled is first interpolated
// linearly on its mean step.
//
// Syntax:   double val = phi.robustness(x, t);
// Inputs:   x - the states of the trace (m, n), one per row
//           t - the times of the trace (m)
// Outputs:  val - positive if the trace satisfies the formula, negative if it violates it
// See also: stl.h
// References:
//    [1] S. Sankaranarayanan et al. "Falsification of temporal properties of hybrid systems using
//        the cross-entropy method", HSCC 2012.
//
// CORA: specification/@stl/robustness.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/specification/stl.h"
#include "specification/private/priv_project_polytope.h"
#include "specification/stl/private/priv_trace.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The regions of a predicate ----------------------------------------------------------------------

constexpr double kInf = std::numeric_limits<double>::infinity();

/// One constraint g(x) <= 0 of a region: linear c x - d, or sign * (e1 - e2) of a nonlinear one.
struct Row {
    bool linear = true;
    std::vector<double> c;
    double d = 0;
    StlExpr e1, e2;
    double sign = 1;
};

/// An unsafe region: the points where all rows hold; polytope regions use the distance to the set.
struct Region {
    std::vector<Row> rows;
    bool level = false; ///< a level set, valued by its largest row
};

/// The linear form c x + d of an expression over the columns names (CORA's aux_getHalfspace).
void aux_linear(const StlExpr &e, const std::vector<std::string> &names, std::vector<double> &c,
                double &d) {
    c.assign(names.size(), 0.0);
    d = 0;
    const StlExprKind k = e.kind();
    if (k == StlExprKind::number) {
        d = e.value();
    } else if (k == StlExprKind::variable) {
        c[static_cast<std::size_t>(std::find(names.begin(), names.end(), e.name()) -
                                   names.begin())] = 1;
    } else if (k == StlExprKind::add || k == StlExprKind::sub) {
        std::vector<double> c2;
        double d2 = 0;
        aux_linear(e.left(), names, c, d);
        aux_linear(e.right(), names, c2, d2);
        const double s = k == StlExprKind::add ? 1 : -1;
        for (std::size_t i = 0; i < c.size(); ++i) c[i] += s * c2[i];
        d += s * d2;
    } else if (k == StlExprKind::mul) {
        // the numeric factor scales the other
        const bool left_number = e.left().kind() == StlExprKind::number;
        const double a = (left_number ? e.left() : e.right()).value();
        aux_linear(left_number ? e.right() : e.left(), names, c, d);
        for (double &v : c) v *= a;
        d *= a;
    } else {
        CORA_FAIL(Error::not_supported, "robustness: the comparison is not linear");
    }
}

/// The comparisons of a conjunction.
void aux_cube(const Stl &clause, std::vector<Stl> &comps) {
    if (clause.kind() == StlKind::comparison) {
        comps.push_back(clause);
    } else if (clause.kind() == StlKind::and_) {
        aux_cube(clause.lhs(), comps);
        aux_cube(clause.rhs(), comps);
    } else {
        CORA_FAIL(Error::not_supported, "robustness: conversion to a set is not supported");
    }
}

/// The safe set of a clause as one region of rows (CORA's convert2set): c x <= d, or the level set.
Region aux_safe_set(const Stl &clause, const std::vector<std::string> &names) {
    std::vector<Stl> comps;
    aux_cube(clause, comps);
    Region R;
    R.level = !clause.is_linear_predicate();
    for (const Stl &s : comps) {
        const bool less = s.cmp() == StlCmp::lt || s.cmp() == StlCmp::le;
        Row row;
        row.linear = !R.level;
        if (R.level) {
            row.e1 = s.expr_lhs(), row.e2 = s.expr_rhs(), row.sign = less ? 1 : -1;
        } else {
            std::vector<double> c2;
            double d1 = 0, d2 = 0;
            aux_linear(s.expr_lhs(), names, row.c, d1);
            aux_linear(s.expr_rhs(), names, c2, d2);
            for (std::size_t i = 0; i < c2.size(); ++i) row.c[i] -= c2[i];
            row.d = d2 - d1;
            // a greater-than reads -c x <= -d
            if (!less) {
                for (double &v : row.c) v = -v;
                row.d = -row.d;
            }
        }
        R.rows.push_back(row);
    }
    return R;
}

/// The regions of the reversed constraints of a safe set (CORA's aux_reverseInequalityConstraints).
std::vector<Region> aux_reverse(const Region &S) {
    std::vector<Region> out;
    for (const Row &row : S.rows) {
        Row r = row;
        if (row.linear) {
            // ~polytope(A, b) is polytope(-A, -b - eps)
            for (double &v : r.c) v = -v;
            r.d = -row.d - machine_eps();
        } else {
            r.sign = -row.sign;
        }
        out.push_back({{r}, S.level});
    }
    return out;
}

/// The rows of a region as a polytope.
CORA_FROM_NUMBERS Polytope aux_polytope(const Region &R, std::size_t n) {
    std::vector<double> A, b;
    for (const Row &row : R.rows) {
        A.insert(A.end(), row.c.begin(), row.c.end());
        b.push_back(row.d);
    }
    const int64_t m = static_cast<int64_t>(b.size());
    return Polytope(Tensor::from_data(A, {m, static_cast<int64_t>(n)}), Tensor::from_data(b, {m}));
}

/// The unsafe regions of a predicate (CORA's aux_preprocessTemporalLogic and aux_safe2unsafe).
std::vector<Region> aux_unsafe(const Stl &pred, const std::vector<std::string> &names) {
    const std::vector<Stl> clauses = pred.disjunctive_normal_form().get_clauses("dnf");
    if (clauses.size() == 1) return aux_reverse(aux_safe_set(clauses[0], names));
    // a union of safe sets: all combinations of one reversed constraint of each
    std::vector<Region> list = aux_reverse(aux_safe_set(clauses[0], names));
    for (std::size_t i = 1; i < clauses.size(); ++i) {
        const std::vector<Region> next = aux_reverse(aux_safe_set(clauses[i], names));
        std::vector<Region> joined;
        for (const Region &b : next)
            for (const Region &a : list) {
                Region both = a;
                both.rows.insert(both.rows.end(), b.rows.begin(), b.rows.end());
                both.level = a.level || b.level;
                // two polytopes that do not meet are dropped
                if (!both.level && !aux_polytope(a, names.size())
                                        .is_intersecting(aux_polytope(b, names.size()), "exact",
                                                         numerics::kStlPolytopeMeet))
                    continue;
                joined.push_back(both);
            }
        list = joined;
    }
    return list;
}

// The robustness of a point -----------------------------------------------------------------------

/// The robustness of a point for one unsafe region (CORA's aux_robustnessPoint).
double aux_point(const Region &R, const std::vector<std::string> &names, const double *p) {
    const std::size_t n = names.size();
    // a level set: the largest value of its rows
    if (R.level) {
        double val = -kInf;
        for (const Row &row : R.rows) {
            double g = 0;
            if (row.linear) {
                for (std::size_t i = 0; i < n; ++i) g += row.c[i] * p[i];
                g -= row.d;
            } else {
                g = row.sign * (priv_eval(row.e1, names, p) - priv_eval(row.e2, names, p));
            }
            val = std::max(val, g);
        }
        return val;
    }
    // a polytope: inside, the signed distance to the nearest facet; else the distance to it
    std::vector<double> len(R.rows.size()), off(R.rows.size());
    bool inside = true;
    for (std::size_t k = 0; k < R.rows.size(); ++k) {
        double s = 0, l = 0;
        for (std::size_t i = 0; i < n; ++i)
            s += R.rows[k].c[i] * p[i], l += R.rows[k].c[i] * R.rows[k].c[i];
        len[k] = std::sqrt(l), off[k] = s - R.rows[k].d;
        inside = inside && off[k] <= machine_eps();
    }
    if (inside) {
        double val = -kInf;
        for (std::size_t k = 0; k < off.size(); ++k) val = std::max(val, off[k] / len[k]);
        return val;
    }
    // parallel rows form a halfspace: the distance to its facet
    bool halfspace = true;
    for (std::size_t k = 1; k < off.size() && halfspace; ++k)
        for (std::size_t i = 0; i < n; ++i)
            if (std::abs(R.rows[k].c[i] / len[k] - R.rows[0].c[i] / len[0]) >
                numerics::kParallelRows)
                halfspace = false;
    if (halfspace) {
        double val = -kInf;
        for (std::size_t k = 0; k < off.size(); ++k) val = std::max(val, off[k] / len[k]);
        return val;
    }
    // else the Euclidean distance to the polytope
    std::vector<double> A, b, q, pt(p, p + n);
    for (const Row &row : R.rows) A.insert(A.end(), row.c.begin(), row.c.end()), b.push_back(row.d);
    if (!priv_project_polytope(A, b, n, pt, q))
        CORA_FAIL(Error::solver_issue, "robustness: the quadratic program is infeasible");
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += (pt[i] - q[i]) * (pt[i] - q[i]);
    return std::sqrt(sum);
}

// The trace on a grid -----------------------------------------------------------------------------

/// The trace on a uniform grid of the mean step (CORA's aux_robustnessTrace interpolation).
StlTrace aux_resample(const StlTrace &tr) {
    const std::size_t m = tr.m;
    double mx = -kInf, sum = 0;
    for (std::size_t i = 1; i < m; ++i)
        mx = std::max(mx, tr.t[i] - tr.t[i - 1]), sum += tr.t[i] - tr.t[i - 1];
    const double dt = sum / static_cast<double>(m - 1);
    if (m < 2 || std::abs(dt - mx) <= machine_eps()) return tr;
    // the samples in increasing time, the first of equal times
    std::vector<std::size_t> order(m);
    for (std::size_t i = 0; i < m; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return tr.t[a] < tr.t[b]; });
    std::vector<std::size_t> uniq;
    for (const std::size_t i : order)
        if (uniq.empty() || tr.t[i] != tr.t[uniq.back()]) uniq.push_back(i);
    if (uniq.size() < 2)
        CORA_FAIL(Error::wrong_value, "robustness: the trace needs two different times");
    // the grid 0:dt:t_end, as MATLAB's colon counts it
    const double tend = tr.t[order.back()];
    std::size_t count = static_cast<std::size_t>(std::floor(tend / dt));
    if ((static_cast<double>(count) + 1) * dt <=
        tend + 3 * machine_eps() * std::max(std::abs(tend), dt))
        ++count;
    StlTrace out;
    out.m = count + 1, out.n = tr.n, out.names = tr.names;
    for (std::size_t k = 0; k < out.m; ++k) {
        const double tq = std::min(static_cast<double>(k) * dt, tend);
        out.t.push_back(tq);
        // the segment of tq, the end segments extrapolate
        std::size_t s = 0;
        while (s + 2 < uniq.size() && tr.t[uniq[s + 1]] <= tq) ++s;
        const double t0 = tr.t[uniq[s]], t1 = tr.t[uniq[s + 1]];
        for (std::size_t j = 0; j < tr.n; ++j) {
            const double y0 = tr.row(uniq[s])[j], y1 = tr.row(uniq[s + 1])[j];
            out.x.push_back(y0 + (tq - t0) * ((y1 - y0) / (t1 - t0)));
        }
    }
    return out;
}

// The temporal operators --------------------------------------------------------------------------

using Row1 = std::vector<double>;

/// The robustness of a temporal formula at every sample (CORA's aux_robustnessTemporalLogic).
Row1 aux_logic(const Stl &phi, const std::vector<Row1> &r, const std::vector<double> &time,
               int depth) {
    if (!phi.temporal()) return r[static_cast<std::size_t>(phi.id() - 1)];
    const StlKind k = phi.kind();
    if (k == StlKind::not_)
        CORA_FAIL(Error::not_supported, "robustness: a negation of a temporal operator");
    Row1 r1 =
        aux_logic(phi.lhs(), r, time, k == StlKind::and_ || k == StlKind::or_ ? depth : depth + 1);
    const std::size_t m = r1.size();
    if (k == StlKind::and_ || k == StlKind::or_) {
        const Row1 r2 = aux_logic(phi.rhs(), r, time, depth);
        for (std::size_t i = 0; i < m; ++i)
            r1[i] = k == StlKind::and_ ? std::min(r1[i], r2[i]) : std::max(r1[i], r2[i]);
        return r1;
    }
    Row1 res(m, -kInf);
    if (k == StlKind::next) {
        // the samples from the time of the shift on, then -inf
        std::size_t out = 0;
        for (std::size_t i = 0; i < m; ++i)
            if (time[i] >= phi.from()) res[out++] = r1[i];
        return res;
    }
    const bool two = k == StlKind::until || k == StlKind::release;
    Row1 r2;
    if (two) r2 = aux_logic(phi.rhs(), r, time, depth + 1);
    if (k == StlKind::release)
        for (std::size_t i = 0; i < m; ++i) r1[i] = -r1[i], r2[i] = -r2[i];
    std::vector<std::size_t> index;
    for (std::size_t i = 0; i < m; ++i)
        if (time[i] >= phi.from() && time[i] <= phi.to()) index.push_back(i);
    // the window slides one sample per start sample cnt; below the top level only the first counts
    for (std::size_t cnt = 0; !index.empty() && index.front() < m; ++cnt) {
        index.erase(
            std::remove_if(index.begin(), index.end(), [&](std::size_t i) { return i >= m; }),
            index.end());
        if (k == StlKind::finally || k == StlKind::globally) {
            double v = k == StlKind::finally ? -kInf : kInf;
            for (const std::size_t i : index)
                v = k == StlKind::finally ? std::max(v, r1[i]) : std::min(v, r1[i]);
            res[cnt] = v;
        } else {
            // until: the best witness of the right operand with the left one holding up to it
            for (const std::size_t i : index) {
                double low = r2[i];
                for (std::size_t j = cnt; j <= i; ++j) low = std::min(low, r1[j]);
                res[cnt] = std::max(res[cnt], low);
            }
        }
        for (std::size_t &i : index) ++i;
        if (depth == 0) break;
    }
    if (k == StlKind::release)
        for (double &v : res)
            if (!std::isinf(v)) v = -v;
    return res;
}

} // namespace


// ===========================================  MAIN  =========================================== //

double Stl::robustness(const Tensor &x, const std::vector<double> &t) const {
    detail::Entry entry;
    const StlTrace tr0 = priv_trace(*this, x, t, "robustness");
    // the formula in negation normal form, its predicates as unsafe regions
    const auto [phi, preds] = negation_normal_form().assign_identifiers();
    std::vector<std::vector<Region>> sets;
    for (const Stl &pred : preds) sets.push_back(aux_unsafe(pred, tr0.names));
    // the robustness of every predicate at every sample
    const StlTrace tr = aux_resample(tr0);
    std::vector<Row1> r(preds.size(), Row1(tr.m));
    for (std::size_t j = 0; j < preds.size(); ++j)
        for (std::size_t i = 0; i < tr.m; ++i) {
            double val = kInf;
            for (const Region &R : sets[j]) val = std::min(val, aux_point(R, tr.names, tr.row(i)));
            r[j][i] = val;
        }
    return aux_logic(phi, r, tr.t, 0)[0];
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
