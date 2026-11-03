// priv_reach_adaptive - adaptive reachability of linear systems with a bound on the error [1]
//
// Steps of adaptive size propagate the zonotope of the affine solution and add the input
// solution and the curvature errors; every step size is chosen so that the accumulating
// (particular solution of U), reduction and non-accumulating (curvature, linear combination)
// errors stay within the bounds that add up to the given error. Steps where every specification
// is verified only propagate the time-point set (verify).
//
// Syntax:   R = priv_reach_adaptive(taylor, params, spec_times, error, verify, savedata);
// Inputs:   taylor - Taylor cache of A;  params - canonical model parameters;
//           spec_times - unverified times per specification (verify);  error - error bound
//           verify - called from verify;  savedata - step sizes and sets of earlier runs
// Outputs:  R - time-interval and time-point output sets with errors, and the time points
// References:
//    [1] M. Wetzlinger et al. "Fully automated verification of linear systems using inner- and
//        outer-approximations of reachable sets", TAC, 2023.
// See also: priv_verify_ra_zonotope

#include "cont_dynamics/linear_sys/private/priv_reach_adaptive.h"
#include "cont_dynamics/linear_sys/private/priv_lin_error_bound.h"
#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kMaxEta = 75; // Taylor terms of the particular solution before it counts as slow

using Cols = std::vector<std::vector<double>>; // the columns of a matrix

/// The auxiliary sets of the running step, and the solution propagated over the steps.
struct SetState {
    // Propagated over the steps.
    std::optional<Zonotope> startset, Hstartp, enc;
    Tensor Pu, Putotal, G_PUtotal_infty; // (n, 1)
    // The particular solution of U, reduced: columns of G, of G without its largest entry, and
    // their girard metric; the box part G_PUtotal_infty is kept as a vector.
    Cols G_PUtotal_zero, Ghat_PUtotal_zero;
    std::vector<double> girard_pu_total_zero;
    // One entry per trial step size of the running step.
    std::vector<Tensor> box_fc, box_fg, GuC, GuG, PUzero, PUinf, PUAsum;
};

/// The values of a chosen trial.
struct Chosen {
    Tensor box_fc, box_fg, GuC, GuG, PUzero, PUinf, PUAsum;
};

// Vectors and matrices ----------------------------------------------------------------------------

/// The 2-norm of a vector.
double aux_vecnorm(const std::vector<double> &v) {
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

/// MATLAB's vecnorm(sum(abs(M), 2)): the 2-norm of the row sums of |M|.
double aux_row_sum_norm(const Tensor &M) { return aux_vecnorm(M.abs().sum(-1, true).data()); }

/// The matrix M (r, c) as its columns.
Cols aux_columns(const Tensor &M) {
    const std::vector<int64_t> s = M.shape();
    const std::vector<double> d = M.data();
    Cols cols(s[1], std::vector<double>(s[0]));
    for (int64_t i = 0; i < s[0]; ++i)
        for (int64_t j = 0; j < s[1]; ++j) cols[j][i] = d[i * s[1] + j];
    return cols;
}

/// The columns as a matrix (n, cols.size()) like `like`.
Tensor aux_from_columns(const Tensor &like, const Cols &cols, int64_t n) {
    const int64_t m = static_cast<int64_t>(cols.size());
    std::vector<double> d(n * m);
    for (int64_t j = 0; j < m; ++j)
        for (int64_t i = 0; i < n; ++i) d[i * m + j] = cols[j][i];
    return Tensor::like(like, d, {n, m});
}

/// The 2-norm of a matrix: the square root of the largest eigenvalue of its Gram matrix, by
/// Jacobi rotations for up to 64 rows or columns, else by the power method.
double aux_norm2(const Tensor &M) {
    const std::vector<int64_t> s = M.shape();
    if (s[0] == 0 || s[1] == 0) return 0.0;
    const Tensor Mt = M.matrix_transpose();
    const bool rows_smaller = s[0] <= s[1];
    const Tensor Gram = rows_smaller ? M.matmul(Mt) : Mt.matmul(M);
    const int64_t d = std::min(s[0], s[1]);
    std::vector<double> a = Gram.data();
    if (d <= 64) {
        // Cyclic Jacobi sweeps until the off-diagonal part vanishes.
        for (int sweep = 0; sweep < 100; ++sweep) {
            double off = 0, diag = 0;
            for (int64_t i = 0; i < d; ++i)
                for (int64_t j = 0; j < d; ++j)
                    (i == j ? diag : off) += a[i * d + j] * a[i * d + j];
            if (off <= numerics::kJacobiOffDiagonal * (diag + numerics::kTiny)) break;
            for (int64_t p = 0; p < d; ++p)
                for (int64_t q = p + 1; q < d; ++q) {
                    if (a[p * d + q] == 0) continue;
                    const double theta = (a[q * d + q] - a[p * d + p]) / (2 * a[p * d + q]);
                    const double t = (theta >= 0 ? 1.0 : -1.0) /
                                     (std::abs(theta) + std::sqrt(theta * theta + 1));
                    const double c = 1 / std::sqrt(t * t + 1), sn = t * c;
                    for (int64_t k = 0; k < d; ++k) {
                        const double akp = a[k * d + p], akq = a[k * d + q];
                        a[k * d + p] = c * akp - sn * akq;
                        a[k * d + q] = sn * akp + c * akq;
                    }
                    for (int64_t k = 0; k < d; ++k) {
                        const double apk = a[p * d + k], aqk = a[q * d + k];
                        a[p * d + k] = c * apk - sn * aqk;
                        a[q * d + k] = sn * apk + c * aqk;
                    }
                }
        }
        // The diagonal now holds the eigenvalues.
        double lmax = 0;
        for (int64_t i = 0; i < d; ++i) lmax = std::max(lmax, a[i * d + i]);
        return std::sqrt(lmax);
    }
    std::vector<double> x(d, 1.0 / std::sqrt(double(d))), y(d);
    double lambda = 0;
    for (int iter = 0; iter < 10000; ++iter) {
        for (int64_t i = 0; i < d; ++i) {
            y[i] = 0;
            for (int64_t j = 0; j < d; ++j) y[i] += a[i * d + j] * x[j];
        }
        const double ny = aux_vecnorm(y);
        if (ny == 0) return 0.0;
        for (int64_t i = 0; i < d; ++i) x[i] = y[i] / ny;
        if (std::abs(ny - lambda) <= numerics::kPowerIteration * ny) {
            lambda = ny;
            break;
        }
        lambda = ny;
    }
    return std::sqrt(lambda);
}


// Particular solutions ----------------------------------------------------------------------------

/// The particular solution of the constant input u over one step, A^-1 (e^{A dt} - I) u or the
/// Taylor series up to floating-point precision.
Tensor aux_particular_constant(TaylorLinSys &taylor, const Tensor &u, double time_step) {
    const Tensor &A = taylor.A();
    if (u.abs().sum().item() == 0) return u.zeros_like();
    if (const std::optional<Tensor> Ainv = taylor.Ainv())
        return Ainv->matmul((taylor.e_adt(time_step) - A.eye_like()).matmul(u));
    Tensor Asum = A.eye_like() * time_step;
    for (int eta = 1; eta <= kMaxEta; ++eta) {
        const Tensor add_term = taylor.Apower(eta) * taylor.dtoverfac(time_step, eta + 1);
        double max_abs = 0, max_sum = 0;
        bool small = true;
        const std::vector<double> add = add_term.data(), sum = Asum.data();
        for (std::size_t i = 0; i < add.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(add[i]));
            small = small && std::abs(add[i]) <= machine_eps() * std::abs(sum[i]);
        }
        (void)max_sum;
        // Too large a step size diverges: the sum has not converged after 75 terms.
        if (std::isinf(max_abs) || eta == kMaxEta)
            CORA_FAIL(Error::special_error,
                      "LinearSys::verify: the particular solution of u diverges");
        if (small) break;
        Asum = Asum + add_term;
    }
    return Asum.matmul(u);
}

/// The particular solution of the uncertain input U over one step and the error terms of its
/// Taylor series: G_zero = dt GU, G_infty the later terms and A_sum_error their matrix sum;
/// false if the series does not converge.
bool aux_pu(TaylorLinSys &taylor, const Tensor &GU, double time_step, Tensor &G_zero,
            Tensor &G_infty, Tensor &A_sum_error) {
    G_zero = GU * time_step;
    std::vector<double> PUdiag = G_zero.abs().sum(-1, true).data();
    std::vector<Tensor> parts;
    A_sum_error = taylor.A().zeros_like();
    for (int eta = 1;; ++eta) {
        const Tensor add_term = taylor.Apower(eta) * taylor.dtoverfac(time_step, eta + 1);
        const Tensor add_g = add_term.matmul(GU);
        const std::vector<double> add_diag = add_g.abs().sum(-1, true).data();
        // A step size that is too large converges too late.
        if (eta == kMaxEta) return false;
        bool stop = true;
        for (std::size_t i = 0; i < PUdiag.size(); ++i) {
            stop = stop && std::abs(add_diag[i]) <= machine_eps() * std::abs(PUdiag[i]);
            PUdiag[i] += add_diag[i];
        }
        A_sum_error = A_sum_error + add_term;
        parts.push_back(add_g);
        if (stop) break;
    }
    G_infty = Tensor::concat(parts, -1);
    return true;
}

// Reduction ---------------------------------------------------------------------------------------

/// Propagates the particular solution of U by one step and reduces it (by the girard metric)
/// within the reduction error bound; returns the error committed by the reduction. The box
/// part G_PUtotal_infty takes everything that is reduced and everything axis-aligned.
double aux_reduce(TaylorLinSys &taylor, double t, SetState &set, const Chosen &ch,
                  double err_bound_red, bool is_u) {
    if (!is_u) return 0.0;
    const int64_t n = taylor.A().shape()[0];
    const Tensor e_atk = *taylor.read_e_adt(t);

    // The part of later terms is always boxed: it adds no error.
    set.G_PUtotal_infty = set.G_PUtotal_infty + e_atk.matmul(ch.PUinf).abs().sum(-1, true);

    // The new part e^{A tk} PU_zero, without its axis-aligned generators (girard metric 0).
    Cols G = aux_columns(e_atk.matmul(ch.PUzero));
    std::vector<double> box_axis(n, 0.0);
    Cols new_g, new_ghat;
    std::vector<double> new_girard;
    for (const std::vector<double> &col : G) {
        double sum_abs = 0, max_abs = 0;
        int64_t max_idx = 0;
        for (int64_t i = 0; i < n; ++i) {
            sum_abs += std::abs(col[i]);
            if (std::abs(col[i]) > max_abs) max_abs = std::abs(col[i]), max_idx = i;
        }
        const double girard = sum_abs - max_abs;
        if (girard == 0) {
            for (int64_t i = 0; i < n; ++i) box_axis[i] += std::abs(col[i]);
            continue;
        }
        std::vector<double> hat(n);
        for (int64_t i = 0; i < n; ++i) hat[i] = std::abs(col[i]) - (i == max_idx ? max_abs : 0.0);
        new_g.push_back(col);
        new_ghat.push_back(hat);
        new_girard.push_back(girard);
    }
    // The axis-aligned generators are boxed; the others join the reducible part.
    set.G_PUtotal_infty = set.G_PUtotal_infty + Tensor::like(set.Pu, box_axis, {n, 1});
    const int nr_new = static_cast<int>(new_g.size());
    for (int j = 0; j < nr_new; ++j) {
        set.G_PUtotal_zero.push_back(new_g[j]);
        set.Ghat_PUtotal_zero.push_back(new_ghat[j]);
        set.girard_pu_total_zero.push_back(new_girard[j]);
    }

    // The first bound: the generator of smallest girard metric alone.
    const std::size_t total = set.girard_pu_total_zero.size();
    if (total == 0) return 0.0;
    const std::size_t min_idx =
        std::min_element(set.girard_pu_total_zero.begin(), set.girard_pu_total_zero.end()) -
        set.girard_pu_total_zero.begin();
    if (2 * aux_vecnorm(set.Ghat_PUtotal_zero[min_idx]) > err_bound_red) return 0.0;

    // Reduce the generators of smallest girard metric while the error stays within the bound.
    std::vector<std::size_t> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return set.girard_pu_total_zero[a] < set.girard_pu_total_zero[b];
    });
    std::vector<double> sum_temp(n, 0.0);
    double err_red = 0;
    std::size_t red_idx = total;
    for (std::size_t j = 0; j < total; ++j) {
        for (int64_t i = 0; i < n; ++i) sum_temp[i] += set.Ghat_PUtotal_zero[order[j]][i];
        const double omega = 2 * aux_vecnorm(sum_temp);
        if (omega > err_bound_red) {
            red_idx = j;
            break;
        }
        err_red = omega;
    }
    // The reduced generators become part of the box.
    std::vector<char> remove(total, 0);
    std::vector<double> box_red(n, 0.0);
    for (std::size_t j = 0; j < red_idx; ++j) {
        remove[order[j]] = 1;
        for (int64_t i = 0; i < n; ++i) box_red[i] += std::abs(set.G_PUtotal_zero[order[j]][i]);
    }
    set.G_PUtotal_infty = set.G_PUtotal_infty + Tensor::like(set.Pu, box_red, {n, 1});
    Cols keep_g, keep_hat;
    std::vector<double> keep_girard;
    for (std::size_t j = 0; j < total; ++j)
        if (!remove[j]) {
            keep_g.push_back(set.G_PUtotal_zero[j]);
            keep_hat.push_back(set.Ghat_PUtotal_zero[j]);
            keep_girard.push_back(set.girard_pu_total_zero[j]);
        }
    set.G_PUtotal_zero = keep_g;
    set.Ghat_PUtotal_zero = keep_hat;
    set.girard_pu_total_zero = keep_girard;
    return err_red;
}

// Errors ------------------------------------------------------------------------------------------

/// What the errors of a step depend on besides the step size.
struct Problem {
    TaylorLinSys *taylor;
    Tensor GU, u;   ///< generators of U (n, mU) and the constant input offset (n, 1)
    bool is_u, isu; ///< whether U has generators, and whether the offset is nonzero
};

/// Sets list[k][it], growing the lists.
template <class T>
void aux_set(std::vector<std::vector<T>> &list, int k, int it, T val) {
    if (static_cast<int>(list.size()) <= k) list.resize(k + 1);
    if (static_cast<int>(list[k].size()) <= it) list[k].resize(it + 1, T());
    list[k][it] = val;
}

/// The curvature error of the state, 2 errOp(F startset), with the box that encloses F startset
/// as a center and the sums of its generators, see [1, Prop. 1].
double aux_eps_f(const Interval &F, const Zonotope &startset, Tensor &box_c, Tensor &box_g) {
    const Tensor Fc = F.center(), Frad = F.rad();
    box_c = Fc.matmul(startset.center());
    // The generators of F startset: F_c G and one axis-aligned generator per dimension.
    const Tensor rad_part =
        Frad.matmul(Tensor::concat({startset.center(), startset.generators()}, -1).abs_sum());
    box_g = Fc.matmul(startset.generators()).abs().sum(-1, true) + rad_part;
    return 2 * aux_vecnorm((box_g + box_c.abs()).data());
}

/// The curvature error of the input, 2 errOp(G u), see [1, Prop. 1].
double aux_eps_g(const Interval &G, const Tensor &u, Tensor &Gc, Tensor &Gbox) {
    Gc = G.center().matmul(u);
    Gbox = G.rad().matmul(u.abs());
    return 2 * aux_vecnorm((Gbox + Gc.abs()).data());
}

/// The index (1-based, 0 if none) of a step size used before, within kTimeStepMatch.
int aux_check_time_step(double time_step, const AdaptiveSaveData &sd) {
    for (std::size_t i = 0; i < sd.time_step.size(); ++i)
        if (std::abs(sd.time_step[i] - time_step) < numerics::kTimeStepMatch)
            return static_cast<int>(i) + 1;
    return 0;
}

/// All errors of the trial step size: accumulating (PU), non-accumulating (linear combination,
/// F, G, time dependency of PU); the sets belonging to them. False if a Taylor sum diverges.
bool aux_errors(const Problem &P, int k, int kIter, LinErrorBound &errs, bool fullcomp,
                SetState &set, double t, double time_step, int idx, const AdaptiveSaveData &sd) {
    TaylorLinSys &taylor = *P.taylor;
    const Tensor e_adtk = taylor.e_adt(time_step);
    const std::size_t need = static_cast<std::size_t>(kIter) + 1;
    for (std::vector<Tensor> *v :
         {&set.box_fc, &set.box_fg, &set.GuC, &set.GuG, &set.PUzero, &set.PUinf, &set.PUAsum})
        if (v->size() < need) v->resize(need);
    const Tensor e_atk = P.is_u ? taylor.e_adt(t) : Tensor();

    // Accumulating error: the particular solution of U.
    if (!P.is_u) {
        aux_set(errs.idv_pu_tkplus1, k, kIter, 0.0);
    } else if (idx == 0) {
        Tensor Gz, Gi, As;
        if (!aux_pu(taylor, P.GU, time_step, Gz, Gi, As)) return false;
        set.PUzero[kIter] = Gz, set.PUinf[kIter] = Gi, set.PUAsum[kIter] = As;
        // The Hausdorff distance between the exact particular solution and the computed one.
        aux_set(errs.idv_pu_tkplus1, k, kIter,
                aux_row_sum_norm(e_atk.matmul(As).matmul(P.GU)) +
                    aux_row_sum_norm(e_atk.matmul(Gi)));
    } else {
        // The step size was used before: read what was computed for it.
        set.PUinf[kIter] = *sd.G_PU_infty[idx - 1];
        set.PUAsum[kIter] = *sd.PU_A_sum_error[idx - 1];
        aux_set(errs.idv_pu_tkplus1, k, kIter,
                aux_row_sum_norm(e_atk.matmul(set.PUAsum[kIter]).matmul(P.GU)) +
                    aux_row_sum_norm(e_atk.matmul(set.PUinf[kIter])));
        set.PUzero[kIter] = P.GU * time_step;
    }

    // Non-accumulating errors: NaN unless the step is computed in full.
    aux_set(errs.idv_f, k, kIter, kNaN);
    aux_set(errs.idv_g, k, kIter, kNaN);
    aux_set(errs.idv_lin_comb, k, kIter, kNaN);
    aux_set(errs.idv_pu_tauk, k, kIter, kNaN);
    aux_set(errs.seq_nonacc, k, kIter, kNaN);
    if (fullcomp) {
        // Error of the linear combination, see [1, Prop. 9].
        const Tensor Gminus = (e_adtk - taylor.A().eye_like()).matmul(set.startset->generators());
        aux_set(errs.idv_lin_comb, k, kIter,
                std::sqrt(double(Gminus.shape()[1])) * aux_norm2(Gminus));

        // Curvature error of the state.
        const std::optional<Interval> F = taylor.F(time_step);
        if (!F) return false;
        aux_set(errs.idv_f, k, kIter,
                aux_eps_f(*F, *set.startset, set.box_fc[kIter], set.box_fg[kIter]));

        // Curvature error of the input.
        const int64_t n = taylor.A().shape()[0];
        set.GuC[kIter] = taylor.A().zeros_like({n, 1});
        set.GuG[kIter] = set.GuC[kIter];
        if (!P.isu) {
            aux_set(errs.idv_g, k, kIter, 0.0);
        } else {
            const std::optional<Interval> G = taylor.G(time_step);
            if (!G) return false;
            if (idx > 0 && sd.fullcomp[idx - 1]) {
                set.GuC[kIter] = *sd.GuTrans_center[idx - 1];
                set.GuG[kIter] = *sd.GuTrans_Gbox[idx - 1];
                aux_set(errs.idv_g, k, kIter, sd.e_g[idx - 1]);
            } else {
                aux_set(errs.idv_g, k, kIter, aux_eps_g(*G, P.u, set.GuC[kIter], set.GuG[kIter]));
            }
        }

        // The time dependency that is lost by adding the particular solution: e^{A tk} PU.
        aux_set(errs.idv_pu_tauk, k, kIter,
                !P.is_u ? 0.0
                        : aux_vecnorm((e_atk.matmul(set.PUzero[kIter]).abs().sum(-1, true) +
                                       e_atk.matmul(set.PUinf[kIter]).abs().sum(-1, true))
                                          .data()));
        aux_set(errs.seq_nonacc, k, kIter,
                errs.idv_lin_comb[k][kIter] + errs.idv_pu_tauk[k][kIter] + errs.idv_f[k][kIter] +
                    errs.idv_g[k][kIter]);
    }
    // The accumulating error of this step; e^{A tk} PU_tauk over-approximates it, but the
    // accumulation to the next step uses the error at tk+1.
    aux_set(errs.step_acc, k, kIter, errs.idv_pu_tkplus1[k][kIter]);
    return true;
}

// Step size ---------------------------------------------------------------------------------------

/// The step size adjusted to one used before (to reuse what was computed for it): the largest
/// used step size in (factor * time_step, time_step), else, between two used ones, their midpoint.
/// idx is the 1-based index in the save data, 0 for none.
double aux_adjust_time_step(int k, double time_step, bool is_u, double max_time_step,
                            const AdaptiveSaveData &sd, int &idx) {
    // The first step searches longer for a good step size.
    const double factor = k == 0 ? 0.95 : (is_u ? 0.75 : 0.80);
    int best = 0;
    for (std::size_t i = 0; i < sd.time_step.size(); ++i)
        if (sd.time_step[i] < time_step && sd.time_step[i] > factor * time_step &&
            (best == 0 || sd.time_step[i] > sd.time_step[best - 1]))
            best = static_cast<int>(i) + 1;
    if (best == 0) {
        idx = 0;
        bool all_ge = true, all_le = true;
        double ub = kInf, lb = -kInf;
        for (double v : sd.time_step) {
            all_ge = all_ge && time_step >= v;
            all_le = all_le && time_step <= v;
            if (v >= time_step) ub = std::min(ub, v);
            if (v <= time_step) lb = std::max(lb, v);
        }
        if (!(all_ge || all_le)) time_step = lb + 0.5 * (ub - lb);
    } else {
        time_step = sd.time_step[best - 1];
        idx = best;
    }
    // The maximum admissible step size may not be exceeded.
    if (time_step > max_time_step) {
        time_step = max_time_step;
        idx = 0;
        for (std::size_t i = 0; i < sd.time_step.size(); ++i)
            if (std::abs(time_step - sd.time_step[i]) < numerics::kTimeRound) {
                idx = static_cast<int>(i) + 1;
                break;
            }
    }
    return time_step;
}

/// Saves what was computed for the chosen trial of a step size for later steps and runs;
/// returns the index (1-based) of the step size in the save data.
int aux_savedata(AdaptiveSaveData &sd, int kIter, bool fullcomp, const Problem &P,
                 const SetState &set, const LinErrorBound &errs, int k, double time_step) {
    int idx = 0;
    for (std::size_t i = 0; i < sd.time_step.size(); ++i)
        if (std::abs(sd.time_step[i] - time_step) < numerics::kTimeRound) {
            idx = static_cast<int>(i) + 1;
            break;
        }
    if (idx == 0) {
        idx = static_cast<int>(sd.time_step.size()) + 1;
        sd.time_step.push_back(time_step);
        sd.fullcomp.push_back(0);
        sd.e_g.push_back(kNaN);
        for (auto *v : {&sd.GuTrans_center, &sd.GuTrans_Gbox, &sd.Pu, &sd.G_PU_zero, &sd.G_PU_infty,
                        &sd.PU_A_sum_error})
            v->push_back(std::nullopt);
    }
    // What the step size needs: the input curvature (full steps) and the particular solution.
    sd.time_step[idx - 1] = time_step;
    sd.fullcomp[idx - 1] = fullcomp;
    if (fullcomp) {
        sd.GuTrans_center[idx - 1] = set.GuC[kIter];
        sd.GuTrans_Gbox[idx - 1] = set.GuG[kIter];
        sd.e_g[idx - 1] = errs.idv_g[k][kIter];
    } else {
        sd.GuTrans_center[idx - 1] = std::nullopt;
        sd.GuTrans_Gbox[idx - 1] = std::nullopt;
        sd.e_g[idx - 1] = kNaN;
    }
    if (P.is_u) {
        sd.G_PU_zero[idx - 1] = set.PUzero[kIter];
        sd.G_PU_infty[idx - 1] = set.PUinf[kIter];
        sd.PU_A_sum_error[idx - 1] = set.PUAsum[kIter];
    } else {
        sd.G_PU_zero[idx - 1] = sd.G_PU_infty[idx - 1] = sd.PU_A_sum_error[idx - 1] = std::nullopt;
    }
    return idx;
}

} // namespace


// ===========================================  MAIN  =========================================== //

AdaptiveResult priv_reach_adaptive(TaylorLinSys &taylor, const AdaptiveParams &params,
                                   const std::vector<VerifyTime> &spec_times, double error,
                                   bool verify, AdaptiveSaveData &savedata) {
    AdaptiveResult out;
    const Tensor &A = taylor.A();
    const int64_t n = A.shape()[0];
    auto column = [&](const std::vector<double> &v) { return Tensor::like(A, v, {n, 1}); };

    // Initialization ------------------------------------------------------------------------------
    Problem P{&taylor, params.GU, params.u_trans, params.GU.shape()[1] > 0,
              params.u_trans.abs().sum().item() != 0};

    // Times where specifications are not yet verified; the horizon ends with the last of them.
    VerifyTime time_spec_unsat;
    double t_final = params.t_final;
    if (verify) {
        time_spec_unsat = VerifyTime::unify(spec_times);
        if (time_spec_unsat.empty())
            CORA_FAIL(Error::report_to_dev,
                      "priv_reach_adaptive: every specification is verified already");
        t_final = time_spec_unsat.final_time();
    }
    double t = 0, time_step = t_final;

    SetState set;
    set.startset = params.R0;
    set.Hstartp = params.R0;
    set.Pu = set.Putotal = set.G_PUtotal_infty = column(std::vector<double>(n, 0.0));

    // The error bound is for the state space: the output equation scales it by ||C||.
    const double err_r2_y = params.C ? aux_norm2(*params.C) : 1.0;
    LinErrorBound errs(error / err_r2_y, t_final);
    errs.use_approx_fun = false; // the first step always bisects: more robust

    // The share of the error for the reduction of the particular solution of U.
    if (!P.is_u) {
        errs.bound_red_max = 0;
        savedata.reductionerrorpercentage = 0.0;
    } else if (savedata.reductionerrorpercentage) {
        errs.bound_red_max = errs.emax * *savedata.reductionerrorpercentage;
    } else {
        errs.compute_error_bound_reduction(A, params.GU);
        savedata.reductionerrorpercentage = errs.bound_red_max / errs.emax;
    }

    auto map_c = [&](const Tensor &M) { return params.C ? params.C->matmul(M) : M; };
    out.time_point.push_back(Zonotope(map_c(params.R0.center()), map_c(params.R0.generators())));
    out.time_point_error.push_back(0.0);
    out.time.push_back(0.0);
    std::vector<char> fullcomp;
    std::vector<double> RcontError, RcontTpError;

    // Steps ---------------------------------------------------------------------------------------
    int k = -1;
    while (t_final - t > numerics::kStepCount) {
        ++k;

        // Where a specification is unverified the step is computed in full, otherwise only the
        // time-point set is propagated; the step may not cross the switch.
        double max_time_step_spec = kInf;
        bool fc = true;
        if (time_spec_unsat.num_intervals() > 0 &&
            !time_spec_unsat.time_until_switch(t, max_time_step_spec, fc)) {
            // Within the tolerance of the last boundary: a sliver of the horizon remains.
            max_time_step_spec = t_final - t;
            fc = false;
        }
        fullcomp.push_back(fc);
        if (fc) set.startset = set.Hstartp;
        const double max_time_step = std::min(t_final - t, max_time_step_spec);
        time_step = std::min(max_time_step, time_step);

        // Trial step sizes until both the accumulating and the non-accumulating error are ok.
        int kIter = -1, kIterChosen = 0;
        std::vector<int> time_step_idx;
        auto clear_trials = [&]() {
            for (std::vector<Tensor> *v : {&set.box_fc, &set.box_fg, &set.GuC, &set.GuG,
                                           &set.PUzero, &set.PUinf, &set.PUAsum})
                v->clear();
        };
        clear_trials();
        while (true) {
            ++kIter;
            errs.next_bounds(time_step, t, k, kIter);
            if (static_cast<int>(time_step_idx.size()) <= kIter) time_step_idx.resize(kIter + 1, 0);
            time_step_idx[kIter] = aux_check_time_step(time_step, savedata);
            const int idx_now = time_step_idx[kIter];
            if (!aux_errors(P, k, kIter, errs, fc, set, t, time_step, idx_now, savedata)) {
                // A Taylor sum diverged: a smaller step size, or the last one that worked.
                if (kIter == 0) {
                    time_step *= 0.1;
                    kIter = -1;
                    clear_trials();
                    continue;
                }
                for (int p = 0; p <= kIter; ++p)
                    if (time_step_idx[p] == time_step_idx[kIter - 1]) {
                        kIterChosen = p;
                        break;
                    }
                time_step = savedata.time_step[time_step_idx[kIterChosen] - 1];
                break;
            }

            // New step sizes are saved for later steps and runs.
            if (time_step_idx[kIter] == 0 || fc != savedata.fullcomp[time_step_idx[kIter] - 1])
                time_step_idx[kIter] =
                    aux_savedata(savedata, kIter, fc, P, set, errs, k, time_step);
            errs.update_coefficients_approx_fun(k, kIter, fc);
            aux_set(errs.bound_acc_ok, k, kIter,
                    static_cast<char>(errs.step_acc[k][kIter] < errs.bound_acc[k][kIter]));
            aux_set(errs.bound_nonacc_ok, k, kIter,
                    static_cast<char>(errs.seq_nonacc[k][kIter] < errs.bound_rem[k][kIter] || !fc));
            errs.update_bisection(k, kIter, P.is_u, time_step);

            // Suitable if both errors are ok and, unless the step is at its maximum, close to
            // the bound (else a larger step size is tried).
            if (errs.bound_acc_ok[k][kIter] && (!fc || errs.bound_nonacc_ok[k][kIter])) {
                if (std::abs(time_step - max_time_step) < numerics::kAtMaxStep) {
                    kIterChosen = kIter;
                    break;
                }
                const bool acc_close = errs.step_acc[k][kIter] > 0.90 * errs.bound_acc[k][kIter];
                const bool rem_close = errs.step_acc[k][kIter] + errs.seq_nonacc[k][kIter] -
                                           errs.idv_pu_tkplus1[k][kIter] >
                                       0.90 * errs.bound_rem[k][kIter];
                if (acc_close || (fc && rem_close)) {
                    kIterChosen = kIter;
                    break;
                }
            }

            // Predict the step size for both bounds, adjusted to ones used before.
            time_step =
                errs.estimate_time_step_size(t, k, kIter, fc, time_step, max_time_step, P.is_u);
            int next_idx = 0;
            time_step =
                aux_adjust_time_step(k, time_step, P.is_u, max_time_step, savedata, next_idx);
            if (static_cast<int>(time_step_idx.size()) <= kIter + 1)
                time_step_idx.resize(kIter + 2, 0);
            time_step_idx[kIter + 1] = next_idx;
            bool reuse = false;
            for (int p = 0; p <= kIter && !reuse; ++p)
                if (time_step_idx[p] == next_idx && errs.bound_acc_ok[k][p] &&
                    (!fc || errs.bound_nonacc_ok[k][p])) {
                    kIterChosen = p;
                    reuse = true;
                }
            if (reuse) break;
        }

        // The propagation matrices at the end of the step.
        const Tensor e_adtk = *taylor.read_e_adt(time_step);
        std::optional<Tensor> e_atkplus1;
        if (P.is_u) {
            e_atkplus1 = e_adtk.matmul(*taylor.read_e_adt(t));
            taylor.insert_e_adt(t + time_step, *e_atkplus1);
        } else if (!fc && std::abs(time_step - max_time_step_spec) < numerics::kTimeStepMatch) {
            e_atkplus1 = taylor.e_adt(t + time_step);
        }

        // Keep the sets of the chosen trial; propagate and reduce the particular solution of U.
        Chosen ch;
        if (fc) {
            ch.box_fc = set.box_fc[kIterChosen];
            ch.box_fg = set.box_fg[kIterChosen];
            ch.GuC = set.GuC[kIterChosen];
            ch.GuG = set.GuG[kIterChosen];
        }
        if (P.is_u) {
            ch.PUzero = set.PUzero[kIterChosen];
            ch.PUinf = set.PUinf[kIterChosen];
            ch.PUAsum = set.PUAsum[kIterChosen];
        }
        if (static_cast<int>(errs.step_red.size()) <= k) errs.step_red.resize(k + 1);
        errs.step_red[k] = aux_reduce(taylor, t, set, ch, errs.bound_red[k][kIterChosen], P.is_u);
        errs.accumulate_errors(k, kIterChosen);
        errs.remove_redundant_values(k, kIterChosen);

        // The particular solution of the constant input is a point: it adds no error.
        const int idx = time_step_idx[kIterChosen];
        if (P.isu) {
            if (!savedata.Pu[idx - 1])
                savedata.Pu[idx - 1] = aux_particular_constant(taylor, P.u, time_step);
            set.Pu = *savedata.Pu[idx - 1];
            set.Putotal = e_adtk.matmul(set.Putotal) + set.Pu;
        }

        // Propagation of the sets -----------------------------------------------------------
        if (!fc) {
            // No sets in a verified step; the time-point set is needed to restart the full one.
            RcontError.push_back(kNaN);
            RcontTpError.push_back(kNaN);
            if (std::abs(time_step - max_time_step_spec) < numerics::kTimeStepMatch)
                set.Hstartp = Zonotope(e_atkplus1->matmul(params.R0.center()) + set.Putotal,
                                       e_atkplus1->matmul(params.R0.generators()));
            out.time_interval.push_back(std::nullopt);
            out.time_interval_error.push_back(kNaN);
            out.time_point.push_back(std::nullopt);
            out.time_point_error.push_back(kNaN);
        } else {
            set.Hstartp = Zonotope(e_adtk.matmul(set.startset->center()) + set.Pu,
                                   e_adtk.matmul(set.startset->generators()));
            set.enc = set.startset->linear_combination(*set.Hstartp);
            double Rc, Rtp;
            errs.full_errors(k, Rc, Rtp);
            RcontError.push_back(Rc);
            RcontTpError.push_back(Rtp);

            // The output sets: homogeneous and particular solutions are mapped by C first.
            const Tensor infty = set.G_PUtotal_infty;
            std::vector<Tensor> Gout{map_c(set.enc->generators())};
            std::vector<Tensor> Gtp{map_c(set.Hstartp->generators())};
            Tensor boxes = ch.box_fg + ch.GuG;
            if (P.is_u) {
                boxes = infty + ch.box_fg + ch.GuG;
                if (!set.G_PUtotal_zero.empty()) {
                    const Tensor Gzero = aux_from_columns(A, set.G_PUtotal_zero, n);
                    Gout.push_back(map_c(Gzero));
                    Gtp.push_back(map_c(Gzero));
                }
                Gtp.push_back(map_c(infty.diag()));
            }
            Gout.push_back(map_c(boxes.diag()));
            out.time_interval.push_back(
                Zonotope(map_c(set.enc->center() + ch.box_fc + ch.GuC), Tensor::concat(Gout, -1)));
            out.time_interval_error.push_back(err_r2_y * Rc);
            out.time_point.push_back(
                Zonotope(map_c(set.Hstartp->center()), Tensor::concat(Gtp, -1)));
            out.time_point_error.push_back(err_r2_y * Rtp);
        }

        // The next step starts where this one ends; the approximation functions are now usable.
        t += time_step;
        out.time.push_back(t);
        errs.use_approx_fun = true;
    }

    // The errors respect their bounds (a failure is a bug).
    if (!errs.check_errors(RcontError, RcontTpError, out.internal_checks_ok))
        CORA_FAIL(Error::report_to_dev, "priv_reach_adaptive: the error bounds were not respected");
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
