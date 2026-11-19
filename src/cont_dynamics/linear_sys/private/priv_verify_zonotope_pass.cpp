// priv_verify_zonotope_pass - one sound pass of verify's zonotope algorithm on a given grid
//
// Every rounded operation is the one of CORALean's definition at nearest, in its order: a sum
// over an index from the last index down (sumUp, sumDown and the products of Interval.matMul nest
// to the right), a fold over a list from its first element (List.foldl), and CORALean's maximum,
// minimum and absolute (so the signs of zeros agree). The products of point matrices are point
// boxes at nearest, so one sum serves both ends; zero terms of a sum are skipped (a sum that
// starts at +0 is never -0, so a zero term changes no bit). Given a Lyapunov weight of A that
// Krylov.lyapunovCertificateFloat accepts (set_weight), the errors are also carried in it and the
// smaller of the two carries is kept (Krylov.carryMinFloat).
//
// Syntax:   VerifyZonotopePass pass(sys);   auto steps = pass.run(specs, grid);
// Inputs:   sys - the system, its initial set and its input set on the host
//           specs - safe and unsafe sets
//           grid - the steps of CORA's tuning
// Outputs:  steps - per step the output set and the flags of the specifications
// See also: priv_verify_zonotope_pass.h, priv_verify_ra_zonotope.cpp
//
// CORA.cpp only: the float definitions of CORALean's LinearSys.verifyZonotopePassFloat,
// verifySpecHoldsFloat, verifyGirardReduceFloat and verifyZonotopeWitnessFloat.
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_verify_zonotope_pass.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"
#include "cora/global/threads.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vec = std::vector<double>;
using Vecs = std::vector<std::vector<double>>;

// Sums, products and norms ------------------------------------------------------------------------

/// step_verifyScaledVecFloat: max(0, max_i x_i / s_i).
double aux_scaled_vec(const Vec &s, const Vec &x) {
    return lean_maximum(0.0,
                        lean_max_over(int64_t(s.size()), [&](int64_t i) { return x[i] / s[i]; }));
}

/// step_verifyScaledNormFloat: max(0, max_j sum_k (|m_jk| s_k) / s_j), row-major (n, n).
double aux_scaled_norm(const Vec &s, const Vec &M) {
    const auto n = int64_t(s.size());
    return lean_maximum(0.0, lean_max_over(n, [&](int64_t j) {
                            return lean_sum(n, [&](int64_t k) {
                                return (lean_absolute(M[std::size_t(j * n + k)]) * s[k]) / s[j];
                            });
                        }));
}

/// L R for row-major L (a, n) and R (n, b): each entry a sum over the inner index from the last
/// down, zero entries of L skipped. Rows in parallel for large products.
Vec aux_matmul(const Vec &L, const Vec &R, int64_t a, int64_t n, int64_t b) {
    Vec out(std::size_t(a * b), 0.0);
    [[maybe_unused]] const int threads = threads_for(a, static_cast<long long>(a) * n * b / 256);
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
    for (int64_t i = 0; i < a; ++i) {
        double *row = &out[std::size_t(i * b)];
        for (int64_t k = n - 1; k >= 0; --k) {
            const double x = L[std::size_t(i * n + k)];
            if (x == 0) continue;
            const double *r = &R[std::size_t(k * b)];
            for (int64_t j = 0; j < b; ++j) row[j] = x * r[j] + row[j];
        }
    }
    return out;
}

/// The midpoint of [lo, hi] and the half-width about it (step_verifyMidFloat).
std::pair<double, double> aux_mid(double lo, double hi) {
    const double mid = lo + lean_half(hi - lo);
    return {mid, lean_maximum(hi - mid, mid - lo)};
}

/// step_verifyMulVecFloat of a point product: the midpoints and half-widths of M x, M (q, n).
std::pair<Vec, Vec> aux_mul_vec(const Vec &M, const Vec &x, int64_t q, int64_t n) {
    Vec mid(q), half(q);
    for (int64_t l = 0; l < q; ++l) {
        double sum = 0;
        for (int64_t k = n - 1; k >= 0; --k) sum = M[std::size_t(l * n + k)] * x[k] + sum;
        std::tie(mid[l], half[l]) = aux_mid(sum, sum);
    }
    return {std::move(mid), std::move(half)};
}

// The carry ---------------------------------------------------------------------------------------

/// The state of the pass at t_k (LinearSys.VerifyZonoState): the carried zonotope <c, G> (n, m)
/// and its scaled error h, Phi ~ e^{A t_k} and its error phi, the input's generators, box and
/// error.
struct State {
    Vec c, G;
    double h = 0;
    Vec Phi;
    double phi = 0;
    Vecs P;
    Vec b;
    double rho = 0;
    double beta = 0, psi = 0;  ///< the errors of the zonotope and of Phi in the Lyapunov weight
    double loc1 = 0, loc2 = 0; ///< the step's local errors (step_verifyZonoLocalFloat), scratch
};

/// step_verifyZonoMagFloat: |c| + sum_j |g_j|.
Vec aux_mag(const State &st, int64_t n, int64_t m) {
    Vec out(n);
    for (int64_t l = 0; l < n; ++l)
        out[l] = lean_absolute(st.c[l]) + lean_sum(m, [&](int64_t j) {
                     return lean_absolute(st.G[std::size_t(l * m + j)]);
                 });
    return out;
}

/// The magnitude row sums of a point matrix X (n, p): sum_j |x_lj|, from the last.
double aux_row_mag(const Vec &X, int64_t l, int64_t p) {
    return lean_sum(p, [&](int64_t j) { return lean_absolute(X[std::size_t(l * p + j)]); });
}

/// step_verifyZonoCarryFloat: the zonotope D <c, G> + w carried to t_{k+1}, D Phi, their errors.
State aux_carry(const Vec &s, const Vec &D, double r, const Vec &w_lo, const Vec &w_hi,
                const State &st, int64_t n, int64_t m) {
    const double norm_e = aux_scaled_norm(s, D) + r; // >= ||e^{A dt}||
    State out = st;
    const Vec Dc = aux_matmul(D, st.c, n, n, 1);
    out.G = aux_matmul(D, st.G, n, n, m);
    Vec gap(n);
    out.c.assign(n, 0.0);
    for (int64_t l = 0; l < n; ++l) {
        const auto [wm, wh] = aux_mid(w_lo[l], w_hi[l]);
        out.c[l] = Dc[l] + wm;
        // the widths of the products are 0 at nearest; the half-width of w stays
        const double cw = out.c[l] - out.c[l];
        gap[l] = (cw + lean_sum(m,
                                [&](int64_t j) {
                                    const double g = out.G[std::size_t(l * m + j)];
                                    return g - g;
                                })) +
                 wh;
    }
    // the zonotope's error at t_{k+1}: the carried one, the ball's and the gap (its local error)
    const double local_z = r * aux_scaled_vec(s, aux_mag(st, n, m));
    const double gap_z = aux_scaled_vec(s, gap);
    out.h = ((norm_e * st.h) + local_z) + gap_z;
    out.loc1 = local_z + gap_z;
    // Phi and its error, the same way
    out.Phi = aux_matmul(D, st.Phi, n, n, n);
    Vec width(out.Phi.size());
    for (std::size_t e = 0; e < width.size(); ++e) width[e] = out.Phi[e] - out.Phi[e];
    const double local_p = r * aux_scaled_norm(s, st.Phi);
    const double width_p = aux_scaled_norm(s, width);
    out.phi = ((norm_e * st.phi) + local_p) + width_p;
    out.loc2 = local_p + width_p;
    return out;
}

/// step_verifyZonoWeightFloat: the errors carried in the weight c, beta' = e^{mu dt} beta +
/// sqrt(b) sqrt(n) max s loc, back in the scales by sqrt(1/a) max 1/s_i; h and phi the smaller of
/// these and the Gershgorin carry (Krylov.carryMinFloat).
void aux_weight(const LyapunovCertificate &c, const Vec &s, double dt, const State &st,
                State &st2) {
    const auto n = int64_t(s.size());
    const double g = std::exp(c.rate * dt);
    double count = 0.0;
    for (int64_t i = 0; i < n; ++i) count = 1.0 + count;
    const double kn =
        (std::sqrt(c.upper) * std::sqrt(count)) * lean_max_over(n, [&](int64_t i) { return s[i]; });
    const double back =
        std::sqrt(1.0 / c.lower) * lean_max_over(n, [&](int64_t i) { return 1.0 / s[i]; });
    const double beta = g * st.beta + kn * st2.loc1;
    const double psi = g * st.psi + kn * st2.loc2;
    st2.h = RestartedBound::carry_min(st2.h, beta * back);
    st2.phi = RestartedBound::carry_min(st2.phi, psi * back);
    st2.beta = beta;
    st2.psi = psi;
}

// The reduction -----------------------------------------------------------------------------------

/// LinearSys.verifyBoxOutFloat: the flagged generators stay, the others join the box b.
Vecs aux_box_out(const Vecs &L, const std::vector<char> &keep, Vec &b) {
    const std::size_t n = b.size();
    Vec out(n, 0.0);
    Vecs stay;
    for (std::size_t g = 0; g < L.size(); ++g) {
        if (keep[g]) {
            stay.push_back(L[g]);
            continue;
        }
        for (std::size_t i = 0; i < n; ++i) out[i] = out[i] + lean_absolute(L[g][i]);
    }
    for (std::size_t i = 0; i < n; ++i) b[i] = b[i] + out[i];
    return stay;
}

/// step_verifyGirardFloat: sum |g_i| - max |g_i|, both folded from the first entry.
double aux_girard(const Vec &g) {
    double sum = 0, mx = 0;
    for (double x : g) sum = sum + lean_absolute(x);
    for (double x : g) mx = lean_maximum(mx, lean_absolute(x));
    return sum - mx;
}

/// step_verifyGirardHatFloat: |g| with its first largest entry set to 0.
Vec aux_girard_hat(const Vec &g) {
    std::size_t idx = 0;
    for (std::size_t i = 1; i < g.size(); ++i)
        if (!(lean_absolute(g[i]) <= lean_absolute(g[idx]))) idx = i;
    Vec out(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) out[i] = i == idx ? 0.0 : lean_absolute(g[i]);
    return out;
}

// The output sets ---------------------------------------------------------------------------------

/// step_verifyZonoOfBoxesFloat: the midpoints of the box of centers and of the boxes of the
/// generators, then P; their half-widths join rad.
VerifyZono aux_of_boxes(const Vec &cen_lo, const Vec &cen_hi, const Vecs &cols_lo,
                        const Vecs &cols_hi, const Vecs &P, const Vec &rad) {
    const std::size_t n = rad.size();
    VerifyZono Z;
    Z.c.resize(n);
    Z.r.resize(n);
    Vec cm_half(n), gh_sum(n, 0.0);
    // the center, then each generator: midpoints, their half-widths summed
    for (std::size_t l = 0; l < n; ++l)
        std::tie(Z.c[l], cm_half[l]) = aux_mid(cen_lo[l], cen_hi[l]);
    for (std::size_t g = 0; g < cols_lo.size(); ++g) {
        Vec mid(n);
        for (std::size_t l = 0; l < n; ++l) {
            const auto [m, hw] = aux_mid(cols_lo[g][l], cols_hi[g][l]);
            mid[l] = m;
            gh_sum[l] = gh_sum[l] + hw;
        }
        Z.G.push_back(std::move(mid));
    }
    // the generators P stay, the half-widths join the box
    for (const Vec &g : P) Z.G.push_back(g);
    for (std::size_t l = 0; l < n; ++l) Z.r[l] = (rad[l] + cm_half[l]) + gh_sum[l];
    return Z;
}

/// step_verifyZonoMapFloat: C T, the box of T as the generators C e_i r_i.
VerifyZono aux_map(const Vec &C, const VerifyZono &T, int64_t q, int64_t n) {
    VerifyZono Y;
    Vec halves;
    auto [cm, ch] = aux_mul_vec(C, T.c, q, n);
    Y.c = std::move(cm);
    // C g for each generator, then the axes of the box as generators
    Vec hsum(q, 0.0);
    auto add = [&](const Vec &x) {
        auto [gm, gh] = aux_mul_vec(C, x, q, n);
        for (int64_t l = 0; l < q; ++l) hsum[l] = hsum[l] + gh[l];
        Y.G.push_back(std::move(gm));
    };
    for (const Vec &g : T.G) add(g);
    for (int64_t i = 0; i < n; ++i) {
        Vec e(n, 0.0);
        e[i] = T.r[i];
        add(e);
    }
    Y.r.resize(q);
    for (int64_t l = 0; l < q; ++l) Y.r[l] = ch[l] + hsum[l];
    return Y;
}

// The Farkas certificates -------------------------------------------------------------------------

/// LinearSys.verifyZonoSupportUpFloat: l'c + sum_j |l'g_j| + |l|'r + w'mu.
double aux_support(const VerifyZono &Z, const Vec &l, const Vec &w) {
    const auto q = int64_t(l.size());
    auto dot = [&](const Vec &a, const Vec &b) {
        return lean_sum(q, [&](int64_t i) { return a[i] * b[i]; });
    };
    double gs = 0;
    for (const Vec &g : Z.G) {
        const double d = dot(l, g);
        gs = gs + lean_maximum(lean_absolute(d), lean_absolute(d));
    }
    Vec mu(q), al(q);
    for (int64_t i = 0; i < q; ++i) {
        double sg = 0;
        for (const Vec &g : Z.G) sg = sg + lean_absolute(g[i]);
        mu[i] = (lean_absolute(Z.c[i]) + sg) + Z.r[i];
        al[i] = lean_absolute(l[i]);
    }
    return ((dot(l, Z.c) + gs) + dot(al, Z.r)) + dot(w, mu);
}

/// step_verifyFarkasFloat: lambda >= 0 and lambda'b < inf_Z (H'lambda)'y.
bool aux_farkas(const VerifyZono &Z, const VerifySpecSet &S, const Vec &lam) {
    if (lam.size() != S.b.size()) return false;
    for (double v : lam)
        if (!(0.0 <= v)) return false;
    const std::size_t q = Z.c.size();
    Vec hi(q, 0.0), neg(q), width(q);
    double lb = 0;
    for (std::size_t k = 0; k < lam.size(); ++k) {
        for (std::size_t i = 0; i < q; ++i) hi[i] = hi[i] + lam[k] * S.a[k][i];
        lb = lb + lam[k] * S.b[k];
    }
    for (std::size_t i = 0; i < q; ++i) {
        neg[i] = -hi[i];
        width[i] = hi[i] - hi[i];
    }
    return lb < -aux_support(Z, neg, width);
}

/// step_verifyCramerFloat: the multipliers of rows i and j that give the planar direction d.
std::optional<Vec> aux_cramer(const VerifySpecSet &S, double d0, double d1, std::size_t i,
                              std::size_t j) {
    auto x = [&](std::size_t k, std::size_t c) { return c < S.a[k].size() ? S.a[k][c] : 0.0; };
    const double det = x(i, 0) * x(j, 1) - x(i, 1) * x(j, 0);
    if (det <= 0 && 0 <= det) return std::nullopt;
    Vec lam(S.b.size(), 0.0);
    lam[i] = (d0 * x(j, 1) - d1 * x(j, 0)) / det;
    lam[j] = (x(i, 0) * d1 - x(i, 1) * d0) / det;
    return lam;
}

/// step_verifyApartFloat: a Farkas certificate from a row alone, or for q = 2 from the normal of
/// a generator as a combination of two rows.
bool aux_apart(const VerifyZono &Z, const VerifySpecSet &S) {
    const std::size_t r = S.b.size();
    for (std::size_t k = 0; k < r; ++k) {
        Vec unit(r, 0.0);
        unit[k] = 1.0;
        if (aux_farkas(Z, S, unit)) return true;
    }
    if (Z.c.size() != 2) return false;
    for (const Vec &g : Z.G)
        for (const auto &[d0, d1] : {std::pair{-g[1], g[0]}, std::pair{g[1], -g[0]}})
            for (std::size_t i = 0; i < r; ++i)
                for (std::size_t j = i + 1; j < r; ++j) {
                    const std::optional<Vec> lam = aux_cramer(S, d0, d1, i, j);
                    if (lam && aux_farkas(Z, S, *lam)) return true;
                }
    return false;
}

// The time-interval set ---------------------------------------------------------------------------

/// step_verifyZonoIntervalFloat: the time-interval set over [t_k, t_{k+1}] from st and st' (the
/// states at its ends): CORA's linComb, F <c, G> and G u0 as boxes, the scaled error as a box.
VerifyZono aux_interval(const Vec &s, const VerifyCorrection &F, const Vec &gu_lo, const Vec &gu_hi,
                        const State &st, const State &st2, int64_t n, int64_t m) {
    const std::size_t nn = std::size_t(n * n);
    Vec Fc(nn), Fr(nn), abs_f(nn);
    for (std::size_t e = 0; e < nn; ++e) {
        std::tie(Fc[e], Fr[e]) = aux_mid(F.inf[e], F.sup[e]);
        abs_f[e] = lean_absolute(Fc[e]) + Fr[e];
    }
    const Vec wc = aux_matmul(Fc, st.c, n, n, 1), WG = aux_matmul(Fc, st.G, n, n, m);
    const Vec mag = aux_mag(st, n, m);
    const double gamma = 1.0 + aux_scaled_norm(s, abs_f); // >= ||(1 - lambda) I + F_tau||
    const double e = (st2.h + gamma * st.h) + st2.rho;
    Vec cen_lo(n), cen_hi(n), rad(n);
    Vecs lo, hi;
    for (int64_t j = 0; j < 2 * m + 1; ++j) {
        lo.emplace_back(n);
        hi.emplace_back(n);
    }
    // per row: the box of centers, the generators of linComb, the radius
    for (int64_t l = 0; l < n; ++l) {
        const double hc = lean_half(st.c[l] + st2.c[l]);
        cen_lo[l] = (hc + wc[l]) + gu_lo[l];
        cen_hi[l] = (hc + wc[l]) + gu_hi[l];
        for (int64_t j = 0; j < m; ++j) {
            const double g = st.G[std::size_t(l * m + j)], g2 = st2.G[std::size_t(l * m + j)];
            lo[j][l] = hi[j][l] = lean_half(g + g2);
            lo[m + j][l] = hi[m + j][l] = lean_half(g - g2);
        }
        lo[2 * m][l] = hi[2 * m][l] = lean_half(st.c[l] - st2.c[l]);
        const double wr =
            lean_sum(n, [&](int64_t k) { return Fr[std::size_t(l * n + k)] * mag[k]; });
        const double wg = lean_sum(m, [&](int64_t j) {
            const double x = WG[std::size_t(l * m + j)];
            return lean_maximum(lean_absolute(x), lean_absolute(x));
        });
        rad[l] = ((st2.b[l] + wg) + wr) + s[l] * e;
    }
    return aux_of_boxes(cen_lo, cen_hi, lo, hi, st2.P, rad);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The specifications and the reduction ------------------------------------------------------------

bool priv_verify_spec_holds(const VerifyZono &Z, const VerifySpecSet &S) {
    if (!S.safe) return aux_apart(Z, S);
    const Vec zero(Z.c.size(), 0.0);
    for (std::size_t k = 0; k < S.b.size(); ++k)
        if (!(aux_support(Z, S.a[k], zero) < S.b[k])) return false;
    return true;
}

Vecs priv_verify_girard_reduce(Vecs P, Vecs added, Vec &b, double budget) {
    // the axis-aligned new generators into the box, the others appended
    std::vector<char> keep(added.size());
    for (std::size_t g = 0; g < added.size(); ++g) keep[g] = !(aux_girard(added[g]) <= 0);
    for (Vec &g : aux_box_out(added, keep, b)) P.push_back(std::move(g));
    // the generators in increasing order of the metric (stable)
    std::vector<double> gir(P.size());
    for (std::size_t g = 0; g < P.size(); ++g) gir[g] = aux_girard(P[g]);
    std::vector<std::size_t> order(P.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t i, std::size_t j) { return gir[i] < gir[j]; });
    // the longest prefix with 4 ||sum of hats||_2^2 <= budget^2 joins the box
    const std::size_t n = b.size();
    Vec S(n, 0.0);
    std::size_t count = 0;
    for (const std::size_t g : order) {
        const Vec hat = aux_girard_hat(P[g]);
        for (std::size_t i = 0; i < n; ++i) S[i] = S[i] + hat[i];
        const double omega = 4.0 * lean_sum(int64_t(n), [&](int64_t i) {
                                 return S[std::size_t(i)] * S[std::size_t(i)];
                             });
        if (!(omega <= budget * budget)) break;
        ++count;
    }
    std::vector<char> stay(P.size(), 1);
    for (std::size_t i = 0; i < count; ++i) stay[order[i]] = 0;
    return aux_box_out(P, stay, b);
}

// The pass ----------------------------------------------------------------------------------------

VerifyZonotopePass::VerifyZonotopePass(VerifyZonoSystem sys)
    : sys_(std::move(sys)), powers_(sys_.A.data(), sys_.n) {
    const int64_t n = sys_.n, k = sys_.k, p = sys_.p;
    s_ = priv_verify_balance_scales(powers_.A(), n);
    for (double v : s_) scales_ok_ = scales_ok_ && 0 < v;
    // [B uc] and [B GU], points at nearest
    u0_.assign(n, 0.0);
    MU_.assign(std::size_t(n * p), 0.0);
    for (int64_t l = 0; l < n; ++l) {
        u0_[l] =
            lean_sum(k, [&](int64_t j) { return sys_.B[std::size_t(l * k + j)] * sys_.uc[j]; });
        for (int64_t c = 0; c < p; ++c)
            MU_[std::size_t(l * p + c)] = lean_sum(k, [&](int64_t j) {
                return sys_.B[std::size_t(l * k + j)] * sys_.GU[std::size_t(j * p + c)];
            });
    }
    // which input parts are nonzero
    for (double v : u0_) has_u_ = has_u_ || !(v <= 0 && 0 <= v);
    for (double v : MU_) has_m_ = has_m_ || !(v <= 0 && 0 <= v);
}

void VerifyZonotopePass::set_weight(std::vector<double> P, std::vector<double> L, double a) {
    cert_ = priv_lyapunov_certificate(SparseMatrix::from_dense(sys_.A), P, L, a);
    weight_ = {std::move(P), std::move(L), a};
}

VerifyZonotopePass::Step &VerifyZonotopePass::step(double dt, bool full) {
    const int64_t n = sys_.n, p = sys_.p;
    auto [it, fresh] = steps_.try_emplace(dt);
    Step &S = it->second;
    if (fresh) {
        S.ball = priv_verify_exp_ball(sys_.A, s_, dt);
        // step_verifyZonoInputFloat: no input, or its solution and the box of G u0
        S.gu_ok = true;
        if (has_u_ || has_m_) {
            S.input = priv_verify_input_solution(powers_, s_, dt, u0_, MU_, p, true);
        } else {
            S.input = VerifyInputSolution{
                Vec(n, 0.0), Vec(n, 0.0), {}, Vec(std::size_t(n * p), 0.0), Vec(n, 0.0)};
        }
        // G u0 as a box, each entry summed from the last index
        S.gu_lo.assign(n, 0.0);
        S.gu_hi.assign(n, 0.0);
        if (has_u_) {
            const std::optional<VerifyCorrection> G = priv_verify_correction_input(powers_, s_, dt);
            S.gu_ok = G.has_value();
            // per row: the products of the box of G with u0, low and high
            for (int64_t l = 0; G && l < n; ++l) {
                double lo = 0, hi = 0;
                for (int64_t j = n - 1; j >= 0; --j) {
                    const double u = u0_[j];
                    if (u == 0) continue;
                    const double a = G->inf[std::size_t(l * n + j)] * u;
                    const double b = G->sup[std::size_t(l * n + j)] * u;
                    lo = lean_minimum(a, b) + lo;
                    hi = lean_maximum(a, b) + hi;
                }
                S.gu_lo[l] = lo;
                S.gu_hi[l] = hi;
            }
        }
    }
    // F, for a full step only
    if (full && !S.F_done) {
        S.F = priv_verify_correction_state(powers_, s_, dt);
        S.F_done = true;
    }
    return S;
}

std::optional<std::vector<std::optional<VerifyZonoStep>>>
VerifyZonotopePass::run(const std::vector<VerifySpecSet> &specs,
                        const std::vector<VerifyGridStep> &grid, std::size_t *failed) {
    const int64_t n = sys_.n, m = sys_.m, p = sys_.p;
    if (failed) *failed = grid.size();
    if (!scales_ok_) return std::nullopt;
    // the state at t = 0: <x0, G0>, Phi = I, no errors, no input
    State st;
    st.c = sys_.x0;
    st.G = sys_.G0;
    st.Phi.assign(std::size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) st.Phi[std::size_t(i * n + i)] = 1.0;
    st.b.assign(n, 0.0);
    std::vector<std::optional<VerifyZonoStep>> out;
    out.reserve(grid.size());
    // step by step: the cached data of the step size, the input, the carry, the output set
    for (const VerifyGridStep &g : grid) {
        if (failed) *failed = out.size();
        if (!(0 < g.dt)) return std::nullopt;
        Step &S = step(g.dt, g.full);
        if (!S.ball || !S.input || !S.gu_ok) return std::nullopt;
        const VerifyInputSolution &I = *S.input;
        // step_verifyZonoInputStepFloat: the orders carried by Phi, the remainder, the reduction
        Vecs added;
        Vec box(n, 0.0), first(n, 0.0), later(n, 0.0), beta_u(n, 0.0);
        for (std::size_t i = 0; i < I.V.size(); ++i) {
            const Vec X = aux_matmul(st.Phi, I.V[i], n, n, p);
            if (i > 0) {
                for (int64_t l = 0; l < n; ++l) later[l] = later[l] + aux_row_mag(X, l, p);
                continue;
            }
            // the first order: its columns are generators, its widths (0 at nearest) the box
            for (int64_t l = 0; l < n; ++l)
                first[l] = lean_sum(p, [&](int64_t j) {
                    const double x = X[std::size_t(l * p + j)];
                    return x - x;
                });
            for (int64_t j = 0; j < p; ++j) {
                Vec col(n);
                for (int64_t l = 0; l < n; ++l) col[l] = X[std::size_t(l * p + j)];
                added.push_back(std::move(col));
            }
        }
        // the box: the orders and the remainder carried by Phi; rho: the varying input's error
        for (int64_t l = 0; l < n; ++l) {
            const double ord = I.V.empty() ? 0.0 : first[l] + later[l];
            const double rem = lean_sum(n, [&](int64_t j) {
                return lean_absolute(st.Phi[std::size_t(l * n + j)]) * I.rho[j];
            });
            box[l] = (st.b[l] + ord) + rem;
            double vs = 0;
            for (const Vec &V : I.V) vs = vs + aux_row_mag(V, l, p);
            beta_u[l] = vs + I.rho[l];
        }
        const double rho = st.rho + st.phi * aux_scaled_vec(s_, beta_u);
        Vecs P = priv_verify_girard_reduce(st.P, std::move(added), box, g.budget);
        // the zonotope, Phi and their errors at t_{k+1}
        const Vec D = S.ball->D.data();
        State st2 = aux_carry(s_, D, S.ball->r, I.w_lo, I.w_hi, st, n, m);
        st2.P = std::move(P);
        st2.b = std::move(box);
        st2.rho = rho;
        // step_verifyZonoWeightedFloat: the weight's carry, if a certificate is accepted
        if (cert_ && cert_->ok) {
            double range = 1.0;
            for (int i = 0; i < 29; ++i) range = range + range;
            if (!(cert_->rate * g.dt <= range)) return std::nullopt;
            aux_weight(*cert_, s_, g.dt, st, st2);
        }
        // a full step: the time-interval set in the outputs, and its flags
        if (!g.full) {
            out.emplace_back(std::nullopt);
        } else {
            if (!S.F) return std::nullopt;
            VerifyZonoStep z;
            z.Y =
                aux_map(sys_.C, aux_interval(s_, *S.F, S.gu_lo, S.gu_hi, st, st2, n, m), sys_.q, n);
            for (const VerifySpecSet &spec : specs)
                z.flags.push_back(priv_verify_spec_holds(z.Y, spec));
            out.emplace_back(std::move(z));
        }
        st = std::move(st2);
    }
    if (failed) *failed = grid.size();
    return out;
}

// The witness -------------------------------------------------------------------------------------

bool VerifyZonotopePass::witness(const VerifySpecSet &S, double t, const Vec &b, const Vec &beta) {
    const int64_t n = sys_.n, m = sys_.m, k = sys_.k, p = sys_.p, q = sys_.q;
    bool ok = 0 < t && scales_ok_;
    for (double v : b) ok = ok && lean_absolute(v) <= 1.0;
    for (double v : beta) ok = ok && lean_absolute(v) <= 1.0;
    // x0 + G0 b and B (uc + GU beta), points at nearest
    Vec x0(n), u(k), U0(n);
    for (int64_t i = 0; i < n; ++i)
        x0[i] = sys_.x0[i] +
                lean_sum(m, [&](int64_t j) { return sys_.G0[std::size_t(i * m + j)] * b[j]; });
    for (int64_t i = 0; i < k; ++i)
        u[i] = sys_.uc[i] +
               lean_sum(p, [&](int64_t j) { return sys_.GU[std::size_t(i * p + j)] * beta[j]; });
    for (int64_t i = 0; i < n; ++i)
        U0[i] = lean_sum(k, [&](int64_t j) { return sys_.B[std::size_t(i * k + j)] * u[j]; });
    // e^{A t} as a ball and the input's solution over [0, t]
    const std::optional<VerifyExpBall> ball = priv_verify_exp_ball(sys_.A, s_, t);
    if (!ball) return false;
    const std::optional<VerifyInputSolution> I =
        priv_verify_input_solution(powers_, s_, t, U0, {}, 0);
    if (!I) return false;
    // x(t) within e r nu s of D x0 + w
    const Vec DX = aux_matmul(ball->D.data(), x0, n, n, 1);
    const double nu = lean_maximum(0.0, lean_max_over(n, [&](int64_t i) {
                                       const double a = lean_absolute(x0[i]);
                                       return lean_maximum(a, a) / s_[i];
                                   }));
    const double e = ball->r * nu;
    // the box of x(t), then C x(t) as a box
    Vec lo(n), hi(n), ylo(q), yhi(q);
    for (int64_t i = 0; i < n; ++i) {
        lo[i] = (DX[i] + I->w_lo[i]) - e * s_[i];
        hi[i] = (DX[i] + I->w_hi[i]) + e * s_[i];
    }
    for (int64_t l = 0; l < q; ++l) {
        double a = 0, c = 0;
        for (int64_t j = n - 1; j >= 0; --j) {
            const double cl = sys_.C[std::size_t(l * n + j)];
            const double x = cl * lo[j], y = cl * hi[j];
            a = lean_minimum(lean_minimum(x, y), lean_minimum(x, y)) + a;
            c = lean_maximum(lean_maximum(x, y), lean_maximum(x, y)) + c;
        }
        ylo[l] = a;
        yhi[l] = c;
    }
    // a'y over the box from above
    auto up = [&](const Vec &a) {
        return lean_sum(q, [&](int64_t i) { return lean_maximum(a[i] * ylo[i], a[i] * yhi[i]); });
    };
    if (!ok) return false;
    if (S.safe) {
        for (std::size_t r = 0; r < S.b.size(); ++r) {
            Vec na(q);
            for (int64_t i = 0; i < q; ++i) na[i] = -S.a[r][i];
            if (S.b[r] < -up(na)) return true;
        }
        return false;
    }
    for (std::size_t r = 0; r < S.b.size(); ++r)
        if (!(up(S.a[r]) <= S.b[r])) return false;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
