// linprog - a dense linear program solver (two-phase simplex), as CORA's CORAlinprog
//
// The program is brought to  min c'y, M y = r, y >= 0  by shifting bounded variables, splitting
// free ones and adding slacks; phase 1 minimizes the artificial variables of a full tableau,
// phase 2 the cost. Dantzig's rule pivots until a run of degenerate pivots, then Bland's rule
// pivots, which cannot cycle.
//
// Syntax:   LinProgResult r = linprog(lp);
// Inputs:   lp - the program: cost, inequality and equality constraints, bounds
// Outputs:  r - status (Optimal, Infeasible, Unbounded), the minimizer x and the value fval
// See also: contDynamics/linear_sys/private/priv_verify_ra_zonotope.cpp
//
// CORA: global/functions/matlab/converter/CORAlinprog.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       30-September-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cora/global/linprog.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kMaxPivots = 200000;

/// How one original variable is written with nonnegative ones: x = offset + s1 y[c1] + s2 y[c2].
struct VarMap {
    double offset = 0;
    int c1 = -1, c2 = -1;
    double s1 = 0, s2 = 0;
};

/// The standard form rows M y = r with the basis column of each row.
struct Standard {
    int m = 0, N = 0;
    std::vector<double> M; // (m, N) row-major
    std::vector<double> r, c;
    std::vector<int> basis;    // slack column with +1 of the row, or -1
    std::vector<char> flipped; // the row was multiplied by -1 (its right-hand side was negative)
};

// Simplex -----------------------------------------------------------------------------------------

/// Pivots the tableau T (rows of width W) on entry (row, col).
void aux_pivot(std::vector<double> &T, int m, int W, int row, int col) {
    const double p = T[std::size_t(row) * W + col];
    for (int j = 0; j < W; ++j) T[std::size_t(row) * W + j] /= p;
    for (int i = 0; i < m; ++i) {
        if (i == row) continue;
        const double f = T[std::size_t(i) * W + col];
        if (f == 0) continue;
        for (int j = 0; j < W; ++j) T[std::size_t(i) * W + j] -= f * T[std::size_t(row) * W + j];
        T[std::size_t(i) * W + col] = 0;
    }
}

/// Simplex pivots on the tableau for the cost `cost` over the columns allowed to enter; the
/// return value is 0 (optimal), 1 (unbounded) or 2 (pivot limit).
int aux_simplex(std::vector<double> &T, std::vector<int> &basis, int m, int W,
                const std::vector<double> &cost, const std::vector<char> &allowed) {
    const int N = W - 1;
    int stall = 0;
    for (int iter = 0; iter < kMaxPivots; ++iter) {
        // Reduced costs r_j = c_j - c_B' T[:, j]; entering column by Dantzig or Bland.
        const bool bland = stall > 100;
        int enter = -1;
        double best = -numerics::kSimplexOptimal;
        for (int j = 0; j < N; ++j) {
            if (!allowed[j]) continue;
            double r = cost[j];
            for (int i = 0; i < m; ++i) r -= cost[basis[i]] * T[std::size_t(i) * W + j];
            if (bland ? r < -numerics::kSimplexOptimal : r < best) {
                best = r;
                enter = j;
                if (bland) break;
            }
        }
        if (enter < 0) return 0;
        // Ratio test; ties go to the larger pivot (Bland: the lowest basis index).
        int leave = -1;
        double min_ratio = kInf;
        for (int i = 0; i < m; ++i) {
            const double a = T[std::size_t(i) * W + enter];
            if (a <= numerics::kSimplexPivot) continue;
            const double ratio = std::max(T[std::size_t(i) * W + N], 0.0) / a;
            if (leave < 0 || ratio < min_ratio - numerics::kSimplexTie * (1 + min_ratio)) {
                min_ratio = ratio;
                leave = i;
            } else if (ratio <= min_ratio + numerics::kSimplexTie * (1 + min_ratio) &&
                       (bland ? basis[i] < basis[leave] : a > T[std::size_t(leave) * W + enter])) {
                min_ratio = std::min(ratio, min_ratio);
                leave = i;
            }
        }
        if (leave < 0) return 1;
        stall = min_ratio < numerics::kSimplexTie ? stall + 1 : 0;
        aux_pivot(T, m, W, leave, enter);
        basis[leave] = enter;
    }
    // the iteration limit is reached
    return 2;
}

// Standard form -----------------------------------------------------------------------------------

/// The standard form of lp and the map of each original variable.
Standard aux_standard_form(const LinProg &lp, std::vector<VarMap> &map) {
    const int n = lp.n;
    const int m_ineq = lp.Aineq.empty() ? 0 : static_cast<int>(lp.bineq.size());
    const int m_eq = lp.Aeq.empty() ? 0 : static_cast<int>(lp.beq.size());
    Standard s;
    map.assign(n, VarMap{});
    int cols = 0;
    std::vector<std::pair<int, double>> bound_rows; // (column, upper bound of the shifted y)
    for (int j = 0; j < n; ++j) {
        const double lo = lp.lb.empty() ? -kInf : lp.lb[j];
        const double hi = lp.ub.empty() ? kInf : lp.ub[j];
        if (lo > hi)
            CORA_FAIL(Error::wrong_value, "linprog: a lower bound exceeds its upper bound");
        if (std::isfinite(lo)) {
            map[j] = {lo, cols, -1, 1.0, 0.0};
            if (std::isfinite(hi)) bound_rows.emplace_back(cols, hi - lo);
            ++cols;
        } else if (std::isfinite(hi)) {
            map[j] = {hi, cols++, -1, -1.0, 0.0};
        } else {
            map[j] = {0.0, cols, cols + 1, 1.0, -1.0};
            cols += 2;
        }
    }
    const int n_var = cols;
    s.m = m_ineq + m_eq + static_cast<int>(bound_rows.size());
    // One slack per inequality and per bound row.
    s.N = n_var + m_ineq + static_cast<int>(bound_rows.size());
    s.M.assign(std::size_t(s.m) * s.N, 0.0);
    s.r.assign(s.m, 0.0);
    s.c.assign(s.N, 0.0);
    s.basis.assign(s.m, -1);
    s.flipped.assign(std::size_t(s.m), 0);
    for (int j = 0; j < n; ++j) {
        const double fj = lp.f.empty() ? 0.0 : lp.f[j];
        if (map[j].c1 >= 0) s.c[map[j].c1] += fj * map[j].s1;
        if (map[j].c2 >= 0) s.c[map[j].c2] += fj * map[j].s2;
    }
    auto fill_row = [&](int row, const double *a, double rhs) {
        for (int j = 0; j < n; ++j) {
            if (map[j].c1 >= 0) s.M[std::size_t(row) * s.N + map[j].c1] += a[j] * map[j].s1;
            if (map[j].c2 >= 0) s.M[std::size_t(row) * s.N + map[j].c2] += a[j] * map[j].s2;
            rhs -= a[j] * map[j].offset;
        }
        s.r[row] = rhs;
    };
    // The rows: inequalities with a slack, equalities, then the upper bounds of shifted variables.
    int row = 0;
    for (int i = 0; i < m_ineq; ++i, ++row) {
        fill_row(row, &lp.Aineq[std::size_t(i) * n], lp.bineq[i]);
        s.M[std::size_t(row) * s.N + n_var + i] = 1.0;
    }
    for (int i = 0; i < m_eq; ++i, ++row) fill_row(row, &lp.Aeq[std::size_t(i) * n], lp.beq[i]);
    for (std::size_t b = 0; b < bound_rows.size(); ++b, ++row) {
        s.M[std::size_t(row) * s.N + bound_rows[b].first] = 1.0;
        s.M[std::size_t(row) * s.N + n_var + m_ineq + b] = 1.0;
        s.r[row] = bound_rows[b].second;
    }
    // A row with a nonnegative right-hand side and its own slack starts with that slack basic.
    for (int i = 0; i < s.m; ++i) {
        if (s.r[i] < 0) {
            for (int j = 0; j < s.N; ++j) s.M[std::size_t(i) * s.N + j] *= -1;
            s.r[i] *= -1;
            s.flipped[std::size_t(i)] = 1;
        }
        for (int j = n_var; j < s.N; ++j)
            if (s.M[std::size_t(i) * s.N + j] == 1.0) {
                bool unique = true;
                for (int k = 0; k < s.m; ++k)
                    if (k != i && s.M[std::size_t(k) * s.N + j] != 0) unique = false;
                if (unique) s.basis[i] = j;
            }
    }
    return s;
}

} // namespace


// ===========================================  MAIN  =========================================== //

LinProgResult linprog(const LinProg &lp) {
    if (lp.n <= 0)
        CORA_FAIL(Error::wrong_value, "linprog: the program needs at least one variable");
    std::vector<VarMap> map;
    const Standard s = aux_standard_form(lp, map);
    LinProgResult result;

    // Tableau ---------------------------------------------------------------------------------
    // Tableau with one artificial column per row that has no basic slack, then the rhs.
    int n_art = 0;
    for (int i = 0; i < s.m; ++i) n_art += s.basis[i] < 0;
    const int W = s.N + n_art + 1;
    std::vector<double> T(std::size_t(std::max(s.m, 1)) * W, 0.0);
    std::vector<int> basis(s.m);
    std::vector<int> start(s.m); // the unit column of each row at the start: B^-1 sits there
    std::vector<double> phase1(s.N + n_art, 0.0);
    int art = s.N;
    for (int i = 0; i < s.m; ++i) {
        for (int j = 0; j < s.N; ++j) T[std::size_t(i) * W + j] = s.M[std::size_t(i) * s.N + j];
        T[std::size_t(i) * W + W - 1] = s.r[i];
        if (s.basis[i] >= 0) {
            basis[i] = s.basis[i];
            start[i] = s.basis[i];
        } else {
            T[std::size_t(i) * W + art] = 1.0;
            phase1[art] = 1.0;
            start[i] = art;
            basis[i] = art++;
        }
    }
    // The basis columns must be unit columns: clear them in the other rows.
    for (int i = 0; i < s.m; ++i) aux_pivot(T, s.m, W, i, basis[i]);

    // Phase 1: drive the artificial variables to zero.
    std::vector<char> all(s.N + n_art, 1);
    if (n_art > 0) {
        if (aux_simplex(T, basis, s.m, W, phase1, all) == 2)
            CORA_FAIL(Error::special_error, "linprog: the simplex method did not finish (phase 1)");
        double infeasibility = 0, scale = 1;
        for (int i = 0; i < s.m; ++i) {
            if (basis[i] >= s.N) infeasibility += std::abs(T[std::size_t(i) * W + W - 1]);
            scale = std::max(scale, std::abs(s.r[i]));
        }
        if (infeasibility > numerics::kSimplexFeasible * scale) return result;
        // Basic artificials at zero leave if a real column replaces them, else the row is dropped.
        for (int i = 0; i < s.m; ++i) {
            if (basis[i] < s.N) continue;
            for (int j = 0; j < s.N; ++j)
                if (std::abs(T[std::size_t(i) * W + j]) > numerics::kSimplexDrive) {
                    aux_pivot(T, s.m, W, i, j);
                    basis[i] = j;
                    break;
                }
        }
    }

    // Phase 2: the cost over the real columns only.
    std::vector<double> cost(s.N + n_art, 0.0);
    std::copy(s.c.begin(), s.c.end(), cost.begin());
    std::vector<char> real(s.N + n_art, 0);
    std::fill(real.begin(), real.begin() + s.N, 1);
    const int status = aux_simplex(T, basis, s.m, W, cost, real);
    if (status == 2)
        CORA_FAIL(Error::special_error, "linprog: the simplex method did not finish (phase 2)");
    if (status == 1) {
        result.status = LinProgResult::Status::unbounded;
        return result;
    }

    // Back to the original variables.
    std::vector<double> y(s.N + n_art, 0.0);
    for (int i = 0; i < s.m; ++i) y[basis[i]] = std::max(T[std::size_t(i) * W + W - 1], 0.0);
    result.x.assign(lp.n, 0.0);
    result.fval = 0;
    for (int j = 0; j < lp.n; ++j) {
        double v = map[j].offset;
        if (map[j].c1 >= 0) v += map[j].s1 * y[map[j].c1];
        if (map[j].c2 >= 0) v += map[j].s2 * y[map[j].c2];
        result.x[j] = v;
        if (!lp.f.empty()) result.fval += lp.f[j] * v;
    }
    // A row's multiplier is -pi (+pi if flipped), pi = c_B' B^-1 read at the starting unit column.
    const int m_ineq = lp.Aineq.empty() ? 0 : static_cast<int>(lp.bineq.size());
    const int m_eq = lp.Aeq.empty() ? 0 : static_cast<int>(lp.beq.size());
    for (int k = 0; k < m_ineq + m_eq; ++k) {
        double pi = 0.0;
        for (int i = 0; i < s.m; ++i) pi += cost[basis[i]] * T[std::size_t(i) * W + start[k]];
        const double lambda = s.flipped[std::size_t(k)] ? pi : -pi;
        (k < m_ineq ? result.dual_ineq : result.dual_eq).push_back(lambda);
    }
    result.status = LinProgResult::Status::optimal;
    return result;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
