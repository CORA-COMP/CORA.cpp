// priv_verify_energy - a checked growth bound kappa e^{rt} of e^{tA} from an energy weight
//
// Every rounded operation is the one of CORALean's definition at nearest, in its order: a sum over
// an index from the last index down (sumUp and sumDown nest to the right), maximum and minimum as
// CORALean's (the later operand where they compare equal), the square root the machine's. The
// matrices are sparse rows: a missing entry is CORALean's zero, and every product with it is a
// signed zero that leaves a sum as it is, so skipping it changes no bit. The factor of
// Ellipsoid.cholFactor is formed column by column over the rows that can be nonzero.
//
// Syntax:   auto w = priv_verify_energy_weight(a, n, r);
//           auto kappa = priv_verify_energy_growth(a, s, n, r);
// Inputs:   a - (n, n) system matrix, row-major
//           s - positive scales of the run
//           r - positive rate
// Outputs:  w - the number of positions and the proposed weight, empty if a has not the form
//           kappa - the factor, empty if the weight is not accepted
// See also: priv_verify_energy.h
//
// CORA.cpp only: the float definitions of CORALean's MatExp.energyWeightFloat and
// MatExp.energyGrowthFloat.
// Authors:       Claude Opus 5.5
// Written:       17-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_energy.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Scalars and sparse rows -------------------------------------------------------------------------

using Row = std::vector<std::pair<int64_t, double>>;

constexpr int kScaleFuel = 64;    // the doublings or halvings of a scale
constexpr int kShifts = 40;       // the shifts a = 2^-k tried, k = 1, ..., 40
constexpr int kMarginHalves = 30; // the factor's margin: 2^-30 of each diagonal entry

/// CORALean's maximum and minimum (Binary64.maximum, Binary64.minimum).
double aux_max(double a, double b) { return a <= b ? b : (b < a ? a : a + b); }
double aux_min(double a, double b) { return a <= b ? a : (b < a ? b : a + b); }
double aux_abs(double a) { return aux_max(a, -a); }
bool aux_is_zero(double x) { return x <= 0 && 0 <= x; }

/// The entry j of a row, 0 if it is missing.
double aux_at(const Row &row, int64_t j) {
    const auto it = std::lower_bound(row.begin(), row.end(), j,
                                     [](const auto &e, int64_t c) { return e.first < c; });
    return (it != row.end() && it->first == j) ? it->second : 0.0;
}

/// sum_k x_k y_k over the columns both rows hold, from the last k down.
double aux_dot(const Row &x, const Row &y) {
    double sum = 0;
    std::size_t i = x.size(), j = y.size();
    while (i > 0 && j > 0) {
        const int64_t a = x[i - 1].first, b = y[j - 1].first;
        if (a == b) {
            sum = x[i - 1].second * y[j - 1].second + sum;
            --i;
            --j;
        } else if (a > b) {
            --i;
        } else {
            --j;
        }
    }
    return sum;
}

/// The rows of a matrix by columns: cols[k] lists the rows i with an entry at k, increasing.
std::vector<std::vector<int64_t>> aux_columns(const EnergyRows &M) {
    std::vector<std::vector<int64_t>> cols(M.size());
    for (std::size_t i = 0; i < M.size(); ++i)
        for (const auto &e : M[i]) cols[std::size_t(e.first)].push_back(int64_t(i));
    return cols;
}

// The scales --------------------------------------------------------------------------------------

/// MatExp.step_energyScaleFloat 64 p 1: the power of two s with s^2 p in [1/2, 2).
double aux_scale(double p) {
    double s = 1;
    for (int k = 0; k < kScaleFuel; ++k) {
        const double w = (s * s) * p;
        if (2.0 <= w)
            s = s * 0.5;
        else if ((2.0 * w) < 1.0)
            s = s + s;
        else
            break;
    }
    return s;
}

// The factor and its check ------------------------------------------------------------------------

/// Ellipsoid.cholFactor 64 of Q (all its entries, the diagonal included), column by column.
EnergyRows aux_chol(const EnergyRows &Q) {
    const int64_t n = int64_t(Q.size());
    // the entries Q_ij below the diagonal by column, and the factor's rows and columns
    std::vector<Row> below(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        for (const auto &e : Q[std::size_t(i)])
            if (e.first < i) below[std::size_t(e.first)].emplace_back(i, e.second);
    EnergyRows L(static_cast<std::size_t>(n));
    std::vector<std::vector<int64_t>> cols(static_cast<std::size_t>(n));
    for (int64_t j = 0; j < n; ++j) {
        // the pivot less the margin and the earlier columns' share; its root from below
        const double qjj = aux_at(Q[std::size_t(j)], j);
        double tau = aux_abs(qjj);
        for (int k = 0; k < kMarginHalves; ++k) tau = tau * 0.5;
        const double piv = (qjj - tau) - aux_dot(L[std::size_t(j)], L[std::size_t(j)]);
        const double root = std::sqrt(piv);
        const double d = (0 < root) ? aux_max(0.0, piv) / root : 0.0;
        // the rows below that can be nonzero: Q's column j and the rows sharing a column with j
        std::vector<int64_t> rows;
        for (const auto &e : below[std::size_t(j)]) rows.push_back(e.first);
        for (const auto &e : L[std::size_t(j)])
            for (int64_t i : cols[std::size_t(e.first)])
                if (i > j) rows.push_back(i);
        // the diagonal entry, then the column below it
        if (d != 0) {
            L[std::size_t(j)].emplace_back(j, d);
            cols[std::size_t(j)].push_back(j);
        }
        if (!(0 < d)) continue;
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        for (int64_t i : rows) {
            const double v =
                (aux_at(Q[std::size_t(i)], j) - aux_dot(L[std::size_t(i)], L[std::size_t(j)])) / d;
            if (v == 0) continue;
            L[std::size_t(i)].emplace_back(j, v);
            cols[std::size_t(j)].push_back(i);
        }
    }
    return L;
}

/// Ellipsoid.psdCheck of X (a point at nearest) with the factor L: X - L L' diagonally dominant.
bool aux_psd_check(const EnergyRows &X, const EnergyRows &L) {
    const std::vector<std::vector<int64_t>> cols = aux_columns(L);
    for (std::size_t i = 0; i < X.size(); ++i) {
        // the columns j where X_ij or (L L')_ij can be nonzero
        std::vector<int64_t> js;
        for (const auto &e : X[i]) js.push_back(e.first);
        for (const auto &e : L[i])
            for (int64_t j : cols[std::size_t(e.first)]) js.push_back(j);
        std::sort(js.begin(), js.end());
        js.erase(std::unique(js.begin(), js.end()), js.end());
        double sum = 0;
        for (std::size_t m = js.size(); m-- > 0;) {
            const int64_t j = js[m];
            if (j == int64_t(i)) continue;
            const double v = aux_at(X[i], j) - aux_dot(L[i], L[std::size_t(j)]);
            sum = aux_max(v, -v) + sum;
        }
        const double lo = aux_at(X[i], int64_t(i)) - aux_dot(L[i], L[i]);
        if (!(sum <= lo)) return false;
    }
    return true;
}

/// MatExp.step_energyPsdFloat h X: the check with the factor of X less the coupling of the first
/// h indices to the others.
bool aux_energy_psd(int64_t h, const EnergyRows &X) {
    EnergyRows Q(X.size());
    for (std::size_t i = 0; i < X.size(); ++i)
        for (const auto &e : X[i])
            if ((int64_t(i) < h) == (e.first < h)) Q[i].push_back(e);
    return aux_psd_check(X, aux_chol(Q));
}

// Products and bounds -----------------------------------------------------------------------------

/// The row i of M with the diagonal entry set to v (inserted if missing).
Row aux_with_diagonal(const Row &row, int64_t i, double v) {
    Row out;
    bool done = false;
    for (const auto &e : row) {
        if (!done && e.first >= i) {
            out.emplace_back(i, v);
            done = true;
            if (e.first == i) continue;
        }
        out.push_back(e);
    }
    if (!done) out.emplace_back(i, v);
    return out;
}

/// CORALean's Krylov.quadBoundUp of the symmetric P: max_i (P_ii + half(sum_{j != i} |P_ij| +
/// |P_ji|)).
double aux_quad_bound_up(const EnergyRows &P) {
    const std::size_t n = P.size();
    std::vector<double> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (std::size_t m = P[i].size(); m-- > 0;) {
            const int64_t j = P[i][m].first;
            if (j == int64_t(i)) continue;
            sum = (aux_abs(P[i][m].second) + aux_abs(aux_at(P[std::size_t(j)], int64_t(i)))) + sum;
        }
        f[i] = aux_at(P[i], int64_t(i)) + sum * 0.5;
    }
    double acc = f[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) acc = aux_max(f[i], acc);
    return acc;
}

/// The product M = P A of sparse rows, each entry summed over the inner index from the last down.
EnergyRows aux_product(const EnergyRows &P, const EnergyRows &A) {
    const std::size_t n = P.size();
    EnergyRows M(n);
    std::vector<double> acc(n, 0.0);
    std::vector<char> hit(n, 0);
    std::vector<int64_t> touched;
    for (std::size_t i = 0; i < n; ++i) {
        // row i: the terms over k from the last down, then the touched columns in order
        touched.clear();
        for (std::size_t m = P[i].size(); m-- > 0;) {
            const auto &[k, p] = P[i][m];
            for (const auto &[j, x] : A[std::size_t(k)]) {
                if (!hit[std::size_t(j)]) {
                    hit[std::size_t(j)] = 1;
                    touched.push_back(j);
                }
                acc[std::size_t(j)] = p * x + acc[std::size_t(j)];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (int64_t j : touched) {
            if (acc[std::size_t(j)] != 0) M[i].emplace_back(j, acc[std::size_t(j)]);
            acc[std::size_t(j)] = 0;
            hit[std::size_t(j)] = 0;
        }
    }
    return M;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The weight --------------------------------------------------------------------------------------

std::optional<EnergyWeight> priv_verify_energy_weight(const std::vector<double> &a, int64_t n,
                                                      double r) {
    auto at = [&](int64_t i, int64_t j) { return a[std::size_t(i * n + j)]; };
    auto zero_row = [&](int64_t i) {
        for (int64_t j = 0; j < n; ++j)
            if (!aux_is_zero(at(i, j))) return false;
        return true;
    };
    // the trailing zero rows z, the positions h, and rows 0..h-1 a single 1 at h + i
    int64_t z = 0;
    while (z < n && zero_row(n - 1 - z)) ++z;
    const int64_t h = (n - z) / 2;
    if (!(0 < h && 2 * h + z == n)) return std::nullopt;
    for (int64_t i = 0; i < h; ++i)
        for (int64_t j = 0; j < n; ++j) {
            const double x = at(i, j);
            if (j == h + i ? !(x <= 1.0 && 1.0 <= x) : !aux_is_zero(x)) return std::nullopt;
        }
    // the masses from K = -A_vq: d_j = d_i K_ij / K_ji, the first i < j coupled both ways
    auto K = [&](int64_t i, int64_t j) { return -at(h + i, j); };
    std::vector<double> d;
    for (int64_t j = 0; j < h; ++j) {
        double dj = 1;
        for (int64_t i = 0; i < j; ++i)
            if (!(aux_is_zero(K(i, j)) || aux_is_zero(K(j, i)))) {
                dj = (d[std::size_t(i)] * K(i, j)) / K(j, i);
                break;
            }
        d.push_back(dj);
    }
    double m = d[0];
    for (double x : d) m = aux_min(m, x);
    for (double x : d)
        if (!(0 < x)) return std::nullopt;
    // beta from DF, F = A_vz; 1 if there is no coupling
    double S = 0;
    for (int64_t i = 0; i < h; ++i)
        for (int64_t k = 0; k < z; ++k) {
            const double x = d[std::size_t(i)] * at(h + i, 2 * h + k);
            S = S + x * x;
        }
    const double q = r + r;
    const double beta0 = (S + S) / ((q * q) * m);
    const double beta = beta0 <= 0 ? 1.0 : beta0;
    // the weight: sym(DK) on the positions, D on the velocities, beta on the inputs
    EnergyWeight w;
    w.h = h;
    w.P.resize(std::size_t(n));
    for (int64_t i = 0; i < h; ++i)
        for (int64_t j = 0; j < h; ++j) {
            const double kij = K(i, j), kji = K(j, i);
            if (aux_is_zero(kij) && aux_is_zero(kji) && i != j) continue;
            const double v = (d[std::size_t(i)] * kij + d[std::size_t(j)] * kji) * 0.5;
            if (v != 0 || i == j) w.P[std::size_t(i)].emplace_back(j, v);
        }
    for (int64_t i = h; i < n; ++i)
        w.P[std::size_t(i)].emplace_back(i, i < 2 * h ? d[std::size_t(i - h)] : beta);
    return w;
}

// The growth --------------------------------------------------------------------------------------

std::optional<double> priv_verify_energy_growth(const std::vector<double> &a,
                                                const std::vector<double> &s, int64_t n, double r) {
    const std::optional<EnergyWeight> w = priv_verify_energy_weight(a, n, r);
    if (!w) return std::nullopt;
    const int64_t h = w->h;
    const EnergyRows &P = w->P;
    // the scales sigma ~ 1/sqrt(P_ii), A~ = Sigma^-1 A Sigma and P~ = Sigma P Sigma
    std::vector<double> sg(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) sg[std::size_t(i)] = aux_scale(aux_at(P[std::size_t(i)], i));
    EnergyRows At(static_cast<std::size_t>(n)), Pt(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const double si = sg[std::size_t(i)];
        for (int64_t j = 0; j < n; ++j) {
            const double x = a[std::size_t(i * n + j)];
            if (x == 0) continue;
            const double v = (x * sg[std::size_t(j)]) / si;
            if (v != 0) At[std::size_t(i)].emplace_back(j, v);
        }
        for (const auto &[j, p] : P[std::size_t(i)]) {
            const double v = (si * p) * sg[std::size_t(j)];
            if (v != 0 || j == i) Pt[std::size_t(i)].emplace_back(j, v);
        }
    }
    // the scales positive and P~ symmetric
    if (!(0 < n)) return std::nullopt;
    for (int64_t i = 0; i < n; ++i) {
        if (!(0 < sg[std::size_t(i)])) return std::nullopt;
        for (const auto &[j, v] : Pt[std::size_t(i)]) {
            const double u = aux_at(Pt[std::size_t(j)], i);
            if (!(v <= u && u <= v)) return std::nullopt;
        }
    }
    // P~ - aI certified, the first a = 2^-k that passes
    std::optional<double> shift;
    double ak = 1;
    for (int k = 1; k <= kShifts && !shift; ++k) {
        ak = ak * 0.5;
        EnergyRows X(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            X[std::size_t(i)] =
                aux_with_diagonal(Pt[std::size_t(i)], i, aux_at(Pt[std::size_t(i)], i) - ak);
        if (aux_energy_psd(h, X)) shift = ak;
    }
    if (!shift) return std::nullopt;
    // Z = 2rP~ - P~A~ - (P~A~)' certified
    const EnergyRows M = aux_product(Pt, At);
    EnergyRows Z(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        std::vector<int64_t> js{i};
        for (const auto &e : Pt[std::size_t(i)]) js.push_back(e.first);
        for (const auto &e : M[std::size_t(i)]) js.push_back(e.first);
        std::sort(js.begin(), js.end());
        js.erase(std::unique(js.begin(), js.end()), js.end());
        for (int64_t j : js) {
            const double p = aux_at(Pt[std::size_t(i)], j);
            const double v =
                ((r * p + r * p) - aux_at(M[std::size_t(i)], j)) - aux_at(M[std::size_t(j)], i);
            if (v != 0 || j == i) Z[std::size_t(i)].emplace_back(j, v);
        }
    }
    // the entries (P~A~)_ji with no entry of row i: the transposed pattern
    for (int64_t j = 0; j < n; ++j)
        for (const auto &[i, mji] : M[std::size_t(j)]) {
            Row &row = Z[std::size_t(i)];
            const auto it = std::lower_bound(row.begin(), row.end(), j,
                                             [](const auto &e, int64_t c) { return e.first < c; });
            if (it != row.end() && it->first == j) continue;
            const double p = aux_at(Pt[std::size_t(i)], j);
            const double v = ((r * p + r * p) - aux_at(M[std::size_t(i)], j)) - mji;
            if (v != 0) row.insert(it, {j, v});
        }
    if (!aux_energy_psd(h, Z)) return std::nullopt;
    // kappa = sqrt(b / a) max_i sigma_i / s_i sqrt(sum_j (s_j / sigma_j)^2)
    const double b = aux_quad_bound_up(Pt);
    double sum = 0;
    for (int64_t j = n; j-- > 0;) {
        const double x = s[std::size_t(j)] / sg[std::size_t(j)];
        sum = x * x + sum;
    }
    double mo = sg[std::size_t(n - 1)] / s[std::size_t(n - 1)];
    for (int64_t i = n - 1; i-- > 0;) mo = aux_max(sg[std::size_t(i)] / s[std::size_t(i)], mo);
    return (std::sqrt(b / *shift) * mo) * std::sqrt(sum);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
