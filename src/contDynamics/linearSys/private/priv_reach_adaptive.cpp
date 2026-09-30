// priv_reach_adaptive - adaptive reachability of linear systems with a bound on the error [1]
//
// Steps of adaptive size propagate the zonotope of the affine solution and add the input
// solution and the curvature errors; every step size is chosen so that the accumulating
// (particular solution of U), reduction and non-accumulating (curvature, linear combination)
// errors stay within the bounds that add up to the given error. Steps where every specification
// is verified only propagate the time-point set (verify).
//
// Syntax:   R = priv_reach_adaptive(taylor, params, specTimes, error, verify, savedata);
// Inputs:   taylor - Taylor cache of A;  params - canonical model parameters;
//           specTimes - unverified times per specification (verify);  error - error bound
//           verify - called from verify;  savedata - step sizes and sets of earlier runs
// Outputs:  R - time-interval and time-point output sets with errors, and the time points
// References:
//    [1] M. Wetzlinger et al. "Fully automated verification of linear systems using inner- and
//        outer-approximations of reachable sets", TAC, 2023.
// See also: priv_verifyRA_zonotope

#include "contDynamics/linearSys/private/priv_linErrorBound.h"
#include "contDynamics/linearSys/private/priv_reach_adaptive.h"
#include "contDynamics/linearSys/private/priv_verify.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kEps = 2.220446049250313e-16;  // as MATLAB's eps
constexpr double kNaN =
    std::numeric_limits<double>::quiet_NaN();
constexpr double kInf =
    std::numeric_limits<double>::infinity();
constexpr int kMaxEta = 75;  // Taylor terms of the particular solution before it counts as slow

using Cols = std::vector<std::vector<double>>;  // the columns of a matrix

/// The auxiliary sets of the running step, and the solution propagated over the steps.
struct SetState {
    // Propagated over the steps.
    std::optional<Zonotope> startset, Hstartp, enc;
    Tensor Pu, Putotal, G_PUtotal_infty;  // (n, 1)
    // The particular solution of U, reduced: columns of G, of G without its largest entry, and
    // their girard metric; the box part G_PUtotal_infty is kept as a vector.
    Cols G_PUtotal_zero, Ghat_PUtotal_zero;
    std::vector<double> girard_PUtotal_zero;
    // One entry per trial step size of the running step.
    std::vector<Tensor> boxFc, boxFG, GuC, GuG, PUzero, PUinf, PUAsum;
};

/// The values of a chosen trial.
struct Chosen {
    Tensor boxFc, boxFG, GuC, GuG, PUzero, PUinf, PUAsum;
};

// Vectors and matrices ----------------------------------------------------------------------------

/// The 2-norm of a vector.
double aux_vecnorm(const std::vector<double> &v) {
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

/// MATLAB's vecnorm(sum(abs(M), 2)): the 2-norm of the row sums of |M|.
double aux_rowSumNorm(const Tensor &M) { return aux_vecnorm(M.abs().sumLast().data()); }

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
Tensor aux_fromColumns(const Tensor &like, const Cols &cols, int64_t n) {
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
    const Tensor Mt = M.transpose();
    const bool rowsSmaller = s[0] <= s[1];
    const Tensor Gram = rowsSmaller ? M.matmul(Mt) : Mt.matmul(M);
    const int64_t d = std::min(s[0], s[1]);
    std::vector<double> a = Gram.data();
    if (d <= 64) {
        // Cyclic Jacobi sweeps until the off-diagonal part vanishes.
        for (int sweep = 0; sweep < 100; ++sweep) {
            double off = 0, diag = 0;
            for (int64_t i = 0; i < d; ++i)
                for (int64_t j = 0; j < d; ++j)
                    (i == j ? diag : off) += a[i * d + j] * a[i * d + j];
            if (off <= 1e-30 * (diag + 1e-300)) break;
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
        if (std::abs(ny - lambda) <= 1e-15 * ny) {
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
Tensor aux_particularConstant(TaylorLinSys &taylor, const Tensor &u, double timeStep) {
    const Tensor &A = taylor.A();
    bool nonzero = false;
    for (double v : u.data()) nonzero = nonzero || v != 0;
    if (!nonzero) return u.zerosLike();
    if (const std::optional<Tensor> Ainv = taylor.Ainv())
        return Ainv->matmul((taylor.eAdt(timeStep) - A.eyeLike()).matmul(u));
    Tensor Asum = A.eyeLike() * timeStep;
    for (int eta = 1; eta <= kMaxEta; ++eta) {
        const Tensor addTerm = taylor.Apower(eta) * taylor.dtoverfac(timeStep, eta + 1);
        double maxAbs = 0, maxSum = 0;
        bool small = true;
        const std::vector<double> add = addTerm.data(), sum = Asum.data();
        for (std::size_t i = 0; i < add.size(); ++i) {
            maxAbs = std::max(maxAbs, std::abs(add[i]));
            small = small && std::abs(add[i]) <= kEps * std::abs(sum[i]);
        }
        (void)maxSum;
        // Too large a step size diverges: the sum has not converged after 75 terms.
        if (std::isinf(maxAbs) || eta == kMaxEta)
            throw std::runtime_error("LinearSys::verify: the particular solution of u diverges");
        if (small) break;
        Asum = Asum + addTerm;
    }
    return Asum.matmul(u);
}

/// The particular solution of the uncertain input U over one step and the error terms of its
/// Taylor series: G_zero = dt GU, G_infty the later terms and A_sum_error their matrix sum;
/// false if the series does not converge.
bool aux_PU(
    TaylorLinSys &taylor, const Tensor &GU, double timeStep, Tensor &G_zero, Tensor &G_infty,
    Tensor &A_sum_error) {
    G_zero = GU * timeStep;
    std::vector<double> PUdiag = G_zero.abs().sumLast().data();
    std::vector<Tensor> parts;
    A_sum_error = taylor.A().zerosLike();
    for (int eta = 1;; ++eta) {
        const Tensor addTerm = taylor.Apower(eta) * taylor.dtoverfac(timeStep, eta + 1);
        const Tensor addG = addTerm.matmul(GU);
        const std::vector<double> addDiag = addG.abs().sumLast().data();
        // A step size that is too large converges too late.
        if (eta == kMaxEta) return false;
        bool stop = true;
        for (std::size_t i = 0; i < PUdiag.size(); ++i) {
            stop = stop && std::abs(addDiag[i]) <= kEps * std::abs(PUdiag[i]);
            PUdiag[i] += addDiag[i];
        }
        A_sum_error = A_sum_error + addTerm;
        parts.push_back(addG);
        if (stop) break;
    }
    G_infty = Tensor::catLast(parts);
    return true;
}


// Reduction ---------------------------------------------------------------------------------------

/// Propagates the particular solution of U by one step and reduces it (by the girard metric)
/// within the reduction error bound; returns the error committed by the reduction. The box
/// part G_PUtotal_infty takes everything that is reduced and everything axis-aligned.
double aux_reduce(
    TaylorLinSys &taylor, double t, SetState &set, const Chosen &ch, double errBoundRed, bool isU) {
    if (!isU) return 0.0;
    const int64_t n = taylor.A().shape()[0];
    const Tensor eAtk = *taylor.readEAdt(t);

    // The part of later terms is always boxed: it adds no error.
    set.G_PUtotal_infty = set.G_PUtotal_infty + eAtk.matmul(ch.PUinf).abs().sumLast();

    // The new part e^{A tk} PU_zero, without its axis-aligned generators (girard metric 0).
    Cols G = aux_columns(eAtk.matmul(ch.PUzero));
    std::vector<double> boxAxis(n, 0.0);
    Cols newG, newGhat;
    std::vector<double> newGirard;
    for (const std::vector<double> &col : G) {
        double sumAbs = 0, maxAbs = 0;
        int64_t maxIdx = 0;
        for (int64_t i = 0; i < n; ++i) {
            sumAbs += std::abs(col[i]);
            if (std::abs(col[i]) > maxAbs) maxAbs = std::abs(col[i]), maxIdx = i;
        }
        const double girard = sumAbs - maxAbs;
        if (girard == 0) {
            for (int64_t i = 0; i < n; ++i) boxAxis[i] += std::abs(col[i]);
            continue;
        }
        std::vector<double> hat(n);
        for (int64_t i = 0; i < n; ++i) hat[i] = std::abs(col[i]) - (i == maxIdx ? maxAbs : 0.0);
        newG.push_back(col);
        newGhat.push_back(hat);
        newGirard.push_back(girard);
    }
    // The axis-aligned generators are boxed; the others join the reducible part.
    set.G_PUtotal_infty = set.G_PUtotal_infty + Tensor::like(set.Pu, boxAxis, {n, 1});
    const int nrNew = static_cast<int>(newG.size());
    for (int j = 0; j < nrNew; ++j) {
        set.G_PUtotal_zero.push_back(newG[j]);
        set.Ghat_PUtotal_zero.push_back(newGhat[j]);
        set.girard_PUtotal_zero.push_back(newGirard[j]);
    }

    // The first bound: the generator of smallest girard metric alone.
    const std::size_t total = set.girard_PUtotal_zero.size();
    if (total == 0) return 0.0;
    const std::size_t minIdx = std::min_element(set.girard_PUtotal_zero.begin(),
                                                set.girard_PUtotal_zero.end()) -
                               set.girard_PUtotal_zero.begin();
    if (2 * aux_vecnorm(set.Ghat_PUtotal_zero[minIdx]) > errBoundRed) return 0.0;

    // Reduce the generators of smallest girard metric while the error stays within the bound.
    std::vector<std::size_t> order(total);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return set.girard_PUtotal_zero[a] < set.girard_PUtotal_zero[b];
    });
    std::vector<double> sumTemp(n, 0.0);
    double errRed = 0;
    std::size_t redIdx = total;
    for (std::size_t j = 0; j < total; ++j) {
        for (int64_t i = 0; i < n; ++i) sumTemp[i] += set.Ghat_PUtotal_zero[order[j]][i];
        const double omega = 2 * aux_vecnorm(sumTemp);
        if (omega > errBoundRed) {
            redIdx = j;
            break;
        }
        errRed = omega;
    }
    // The reduced generators become part of the box.
    std::vector<char> remove(total, 0);
    std::vector<double> boxRed(n, 0.0);
    for (std::size_t j = 0; j < redIdx; ++j) {
        remove[order[j]] = 1;
        for (int64_t i = 0; i < n; ++i) boxRed[i] += std::abs(set.G_PUtotal_zero[order[j]][i]);
    }
    set.G_PUtotal_infty = set.G_PUtotal_infty + Tensor::like(set.Pu, boxRed, {n, 1});
    Cols keepG, keepHat;
    std::vector<double> keepGirard;
    for (std::size_t j = 0; j < total; ++j)
        if (!remove[j]) {
            keepG.push_back(set.G_PUtotal_zero[j]);
            keepHat.push_back(set.Ghat_PUtotal_zero[j]);
            keepGirard.push_back(set.girard_PUtotal_zero[j]);
        }
    set.G_PUtotal_zero = keepG;
    set.Ghat_PUtotal_zero = keepHat;
    set.girard_PUtotal_zero = keepGirard;
    return errRed;
}


// Errors ------------------------------------------------------------------------------------------

/// What the errors of a step depend on besides the step size.
struct Problem {
    TaylorLinSys *taylor;
    Tensor GU, u;     ///< generators of U (n, mU) and the constant input offset (n, 1)
    bool isU, isu;    ///< whether U has generators, and whether the offset is nonzero
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
double aux_epsF(const Interval &F, const Zonotope &startset, Tensor &boxC, Tensor &boxG) {
    const Tensor Fc = F.center(), Frad = F.rad();
    boxC = Fc.matmul(startset.c);
    // The generators of F startset: F_c G and one axis-aligned generator per dimension.
    const Tensor radPart = Frad.matmul(Tensor::catLast({startset.c, startset.G}).abs().sumLast());
    boxG = Fc.matmul(startset.G).abs().sumLast() + radPart;
    return 2 * aux_vecnorm((boxG + boxC.abs()).data());
}

/// The curvature error of the input, 2 errOp(G u), see [1, Prop. 1].
double aux_epsG(const Interval &G, const Tensor &u, Tensor &Gc, Tensor &Gbox) {
    Gc = G.center().matmul(u);
    Gbox = G.rad().matmul(u.abs());
    return 2 * aux_vecnorm((Gbox + Gc.abs()).data());
}

/// The index (1-based, 0 if none) of a step size used before, within 1e-9.
int aux_checkTimeStep(double timeStep, const AdaptiveSaveData &sd) {
    for (std::size_t i = 0; i < sd.timeStep.size(); ++i)
        if (std::abs(sd.timeStep[i] - timeStep) < 1e-9) return static_cast<int>(i) + 1;
    return 0;
}

/// All errors of the trial step size: accumulating (PU), non-accumulating (linear combination,
/// F, G, time dependency of PU); the sets belonging to them. False if a Taylor sum diverges.
bool aux_errors(
    const Problem &P, int k, int kIter, LinErrorBound &errs, bool fullcomp, SetState &set, double
    t, double timeStep, int idx, const AdaptiveSaveData &sd) {
    TaylorLinSys &taylor = *P.taylor;
    const Tensor eAdtk = taylor.eAdt(timeStep);
    const std::size_t need = static_cast<std::size_t>(kIter) + 1;
    for (std::vector<Tensor> *v : {&set.boxFc, &set.boxFG, &set.GuC, &set.GuG, &set.PUzero,
                                   &set.PUinf, &set.PUAsum})
        if (v->size() < need) v->resize(need);
    const Tensor eAtk = P.isU ? taylor.eAdt(t) : Tensor();

    // Accumulating error: the particular solution of U.
    if (!P.isU) {
        aux_set(errs.idv_PUtkplus1, k, kIter, 0.0);
    } else if (idx == 0) {
        Tensor Gz, Gi, As;
        if (!aux_PU(taylor, P.GU, timeStep, Gz, Gi, As)) return false;
        set.PUzero[kIter] = Gz, set.PUinf[kIter] = Gi, set.PUAsum[kIter] = As;
        // The Hausdorff distance between the exact particular solution and the computed one.
        aux_set(errs.idv_PUtkplus1, k, kIter,
                aux_rowSumNorm(eAtk.matmul(As).matmul(P.GU)) + aux_rowSumNorm(eAtk.matmul(Gi)));
    } else {
        // The step size was used before: read what was computed for it.
        set.PUinf[kIter] = *sd.G_PU_infty[idx - 1];
        set.PUAsum[kIter] = *sd.PU_A_sum_error[idx - 1];
        aux_set(errs.idv_PUtkplus1, k, kIter,
                aux_rowSumNorm(eAtk.matmul(set.PUAsum[kIter]).matmul(P.GU)) +
                    aux_rowSumNorm(eAtk.matmul(set.PUinf[kIter])));
        set.PUzero[kIter] = P.GU * timeStep;
    }

    // Non-accumulating errors: NaN unless the step is computed in full.
    aux_set(errs.idv_F, k, kIter, kNaN);
    aux_set(errs.idv_G, k, kIter, kNaN);
    aux_set(errs.idv_linComb, k, kIter, kNaN);
    aux_set(errs.idv_PUtauk, k, kIter, kNaN);
    aux_set(errs.seq_nonacc, k, kIter, kNaN);
    if (fullcomp) {
        // Error of the linear combination, see [1, Prop. 9].
        const Tensor Gminus = (eAdtk - taylor.A().eyeLike()).matmul(set.startset->G);
        aux_set(errs.idv_linComb, k, kIter,
                std::sqrt(double(Gminus.shape()[1])) * aux_norm2(Gminus));

        // Curvature error of the state.
        const std::optional<Interval> F = taylor.F(timeStep);
        if (!F) return false;
        aux_set(errs.idv_F, k, kIter,
                aux_epsF(*F, *set.startset, set.boxFc[kIter], set.boxFG[kIter]));

        // Curvature error of the input.
        const int64_t n = taylor.A().shape()[0];
        set.GuC[kIter] = Tensor::like(taylor.A(), std::vector<double>(n, 0.0), {n, 1});
        set.GuG[kIter] = set.GuC[kIter];
        if (!P.isu) {
            aux_set(errs.idv_G, k, kIter, 0.0);
        } else {
            const std::optional<Interval> G = taylor.G(timeStep);
            if (!G) return false;
            if (idx > 0 && sd.fullcomp[idx - 1]) {
                set.GuC[kIter] = *sd.GuTrans_center[idx - 1];
                set.GuG[kIter] = *sd.GuTrans_Gbox[idx - 1];
                aux_set(errs.idv_G, k, kIter, sd.eG[idx - 1]);
            } else {
                aux_set(errs.idv_G, k, kIter, aux_epsG(*G, P.u, set.GuC[kIter], set.GuG[kIter]));
            }
        }

        // The time dependency that is lost by adding the particular solution: e^{A tk} PU.
        aux_set(errs.idv_PUtauk, k, kIter,
                !P.isU ? 0.0
                       : aux_vecnorm((eAtk.matmul(set.PUzero[kIter]).abs().sumLast() +
                                      eAtk.matmul(set.PUinf[kIter]).abs().sumLast()).data()));
        aux_set(errs.seq_nonacc, k, kIter,
                errs.idv_linComb[k][kIter] + errs.idv_PUtauk[k][kIter] + errs.idv_F[k][kIter] +
                    errs.idv_G[k][kIter]);
    }
    // The accumulating error of this step; e^{A tk} PU_tauk over-approximates it, but the
    // accumulation to the next step uses the error at tk+1.
    aux_set(errs.step_acc, k, kIter, errs.idv_PUtkplus1[k][kIter]);
    return true;
}


// Step size ---------------------------------------------------------------------------------------

/// The step size adjusted to one used before (to reuse what was computed for it): the largest
/// used step size in (factor * timeStep, timeStep), else, between two used ones, their midpoint.
/// idx is the 1-based index in the save data, 0 for none.
double aux_adjustTimeStep(
    int k, double timeStep, bool isU, double maxTimeStep, const AdaptiveSaveData &sd, int &idx) {
    // The first step searches longer for a good step size.
    const double factor = k == 0 ? 0.95 : (isU ? 0.75 : 0.80);
    int best = 0;
    for (std::size_t i = 0; i < sd.timeStep.size(); ++i)
        if (sd.timeStep[i] < timeStep && sd.timeStep[i] > factor * timeStep &&
            (best == 0 || sd.timeStep[i] > sd.timeStep[best - 1]))
            best = static_cast<int>(i) + 1;
    if (best == 0) {
        idx = 0;
        bool allGE = true, allLE = true;
        double ub = kInf, lb = -kInf;
        for (double v : sd.timeStep) {
            allGE = allGE && timeStep >= v;
            allLE = allLE && timeStep <= v;
            if (v >= timeStep) ub = std::min(ub, v);
            if (v <= timeStep) lb = std::max(lb, v);
        }
        if (!(allGE || allLE)) timeStep = lb + 0.5 * (ub - lb);
    } else {
        timeStep = sd.timeStep[best - 1];
        idx = best;
    }
    // The maximum admissible step size may not be exceeded.
    if (timeStep > maxTimeStep) {
        timeStep = maxTimeStep;
        idx = 0;
        for (std::size_t i = 0; i < sd.timeStep.size(); ++i)
            if (std::abs(timeStep - sd.timeStep[i]) < 1e-14) {
                idx = static_cast<int>(i) + 1;
                break;
            }
    }
    return timeStep;
}

/// Saves what was computed for the chosen trial of a step size for later steps and runs;
/// returns the index (1-based) of the step size in the save data.
int aux_savedata(
    AdaptiveSaveData &sd, int kIter, bool fullcomp, const Problem &P, const SetState &set, const
    LinErrorBound &errs, int k, double timeStep) {
    int idx = 0;
    for (std::size_t i = 0; i < sd.timeStep.size(); ++i)
        if (std::abs(sd.timeStep[i] - timeStep) < 1e-14) {
            idx = static_cast<int>(i) + 1;
            break;
        }
    if (idx == 0) {
        idx = static_cast<int>(sd.timeStep.size()) + 1;
        sd.timeStep.push_back(timeStep);
        sd.fullcomp.push_back(0);
        sd.eG.push_back(kNaN);
        for (auto *v : {&sd.GuTrans_center, &sd.GuTrans_Gbox, &sd.Pu, &sd.G_PU_zero,
                        &sd.G_PU_infty, &sd.PU_A_sum_error})
            v->push_back(std::nullopt);
    }
    // What the step size needs: the input curvature (full steps) and the particular solution.
    sd.timeStep[idx - 1] = timeStep;
    sd.fullcomp[idx - 1] = fullcomp;
    if (fullcomp) {
        sd.GuTrans_center[idx - 1] = set.GuC[kIter];
        sd.GuTrans_Gbox[idx - 1] = set.GuG[kIter];
        sd.eG[idx - 1] = errs.idv_G[k][kIter];
    } else {
        sd.GuTrans_center[idx - 1] = std::nullopt;
        sd.GuTrans_Gbox[idx - 1] = std::nullopt;
        sd.eG[idx - 1] = kNaN;
    }
    if (P.isU) {
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
                                   const std::vector<VerifyTime> &specTimes, double error,
                                   bool verify, AdaptiveSaveData &savedata) {
    AdaptiveResult out;
    const Tensor &A = taylor.A();
    const int64_t n = A.shape()[0];
    auto column = [&](const std::vector<double> &v) { return Tensor::like(A, v, {n, 1}); };

    // Initialization ------------------------------------------------------------------------------
    Problem P{&taylor, params.GU, params.uTrans, params.GU.shape()[1] > 0, false};
    for (double v : params.uTrans.data()) P.isu = P.isu || v != 0;

    // Times where specifications are not yet verified; the horizon ends with the last of them.
    VerifyTime timeSpecUnsat;
    double tFinal = params.tFinal;
    if (verify) {
        timeSpecUnsat = VerifyTime::unify(specTimes);
        if (timeSpecUnsat.empty())
            throw std::logic_error("priv_reach_adaptive: every specification is verified already");
        tFinal = timeSpecUnsat.finalTime();
    }
    double t = 0, timeStep = tFinal;

    SetState set;
    set.startset = params.R0;
    set.Hstartp = params.R0;
    set.Pu = set.Putotal = set.G_PUtotal_infty = column(std::vector<double>(n, 0.0));

    // The error bound is for the state space: the output equation scales it by ||C||.
    const double errR2Y = params.C ? aux_norm2(*params.C) : 1.0;
    LinErrorBound errs(error / errR2Y, tFinal);
    errs.useApproxFun = false;  // the first step always bisects: more robust

    // The share of the error for the reduction of the particular solution of U.
    if (!P.isU) {
        errs.bound_red_max = 0;
        savedata.reductionerrorpercentage = 0.0;
    } else if (savedata.reductionerrorpercentage) {
        errs.bound_red_max = errs.emax * *savedata.reductionerrorpercentage;
    } else {
        errs.computeErrorBoundReduction(A, params.GU);
        savedata.reductionerrorpercentage = errs.bound_red_max / errs.emax;
    }

    auto mapC = [&](const Tensor &M) { return params.C ? params.C->matmul(M) : M; };
    out.timePoint.push_back(Zonotope(mapC(params.R0.c), mapC(params.R0.G)));
    out.timePointError.push_back(0.0);
    out.time.push_back(0.0);
    std::vector<char> fullcomp;
    std::vector<double> RcontError, RcontTpError;

    // Steps ---------------------------------------------------------------------------------------
    int k = -1;
    while (tFinal - t > 1e-9) {
        ++k;

        // Where a specification is unverified the step is computed in full, otherwise only the
        // time-point set is propagated; the step may not cross the switch.
        double maxTimeStepSpec = kInf;
        bool fc = true;
        if (timeSpecUnsat.numIntervals() > 0 &&
            !timeSpecUnsat.timeUntilSwitch(t, maxTimeStepSpec, fc)) {
            // Within the tolerance of the last boundary: a sliver of the horizon remains.
            maxTimeStepSpec = tFinal - t;
            fc = false;
        }
        fullcomp.push_back(fc);
        if (fc) set.startset = set.Hstartp;
        const double maxTimeStep = std::min(tFinal - t, maxTimeStepSpec);
        timeStep = std::min(maxTimeStep, timeStep);

        // Trial step sizes until both the accumulating and the non-accumulating error are ok.
        int kIter = -1, kIterChosen = 0;
        std::vector<int> timeStepIdx;
        auto clearTrials = [&]() {
            for (std::vector<Tensor> *v : {&set.boxFc, &set.boxFG, &set.GuC, &set.GuG,
                                           &set.PUzero, &set.PUinf, &set.PUAsum})
                v->clear();
        };
        clearTrials();
        while (true) {
            ++kIter;
            errs.nextBounds(timeStep, t, k, kIter);
            if (static_cast<int>(timeStepIdx.size()) <= kIter) timeStepIdx.resize(kIter + 1, 0);
            timeStepIdx[kIter] = aux_checkTimeStep(timeStep, savedata);
            const int idxNow = timeStepIdx[kIter];
            if (!aux_errors(P, k, kIter, errs, fc, set, t, timeStep, idxNow, savedata)) {
                // A Taylor sum diverged: a smaller step size, or the last one that worked.
                if (kIter == 0) {
                    timeStep *= 0.1;
                    kIter = -1;
                    clearTrials();
                    continue;
                }
                for (int p = 0; p <= kIter; ++p)
                    if (timeStepIdx[p] == timeStepIdx[kIter - 1]) {
                        kIterChosen = p;
                        break;
                    }
                timeStep = savedata.timeStep[timeStepIdx[kIterChosen] - 1];
                break;
            }

            // New step sizes are saved for later steps and runs.
            if (timeStepIdx[kIter] == 0 || fc != savedata.fullcomp[timeStepIdx[kIter] - 1])
                timeStepIdx[kIter] = aux_savedata(savedata, kIter, fc, P, set, errs, k, timeStep);
            errs.updateCoefficientsApproxFun(k, kIter, fc);
            aux_set(errs.bound_acc_ok, k, kIter,
                    static_cast<char>(errs.step_acc[k][kIter] < errs.bound_acc[k][kIter]));
            aux_set(errs.bound_nonacc_ok, k, kIter,
                    static_cast<char>(errs.seq_nonacc[k][kIter] < errs.bound_rem[k][kIter] || !fc));
            errs.updateBisection(k, kIter, P.isU, timeStep);

            // Suitable if both errors are ok and, unless the step is at its maximum, close to
            // the bound (else a larger step size is tried).
            if (errs.bound_acc_ok[k][kIter] && (!fc || errs.bound_nonacc_ok[k][kIter])) {
                if (std::abs(timeStep - maxTimeStep) < 1e-12) {
                    kIterChosen = kIter;
                    break;
                }
                const bool accClose = errs.step_acc[k][kIter] > 0.90 * errs.bound_acc[k][kIter];
                const bool remClose = errs.step_acc[k][kIter] + errs.seq_nonacc[k][kIter] -
                                          errs.idv_PUtkplus1[k][kIter] >
                                      0.90 * errs.bound_rem[k][kIter];
                if (accClose || (fc && remClose)) {
                    kIterChosen = kIter;
                    break;
                }
            }

            // Predict the step size for both bounds, adjusted to ones used before.
            timeStep = errs.estimateTimeStepSize(t, k, kIter, fc, timeStep, maxTimeStep, P.isU);
            int nextIdx = 0;
            timeStep = aux_adjustTimeStep(k, timeStep, P.isU, maxTimeStep, savedata, nextIdx);
            if (static_cast<int>(timeStepIdx.size()) <= kIter + 1) timeStepIdx.resize(kIter + 2, 0);
            timeStepIdx[kIter + 1] = nextIdx;
            bool reuse = false;
            for (int p = 0; p <= kIter && !reuse; ++p)
                if (timeStepIdx[p] == nextIdx && errs.bound_acc_ok[k][p] &&
                    (!fc || errs.bound_nonacc_ok[k][p])) {
                    kIterChosen = p;
                    reuse = true;
                }
            if (reuse) break;
        }

        // The propagation matrices at the end of the step.
        const Tensor eAdtk = *taylor.readEAdt(timeStep);
        std::optional<Tensor> eAtkplus1;
        if (P.isU) {
            eAtkplus1 = eAdtk.matmul(*taylor.readEAdt(t));
            taylor.insertEAdt(t + timeStep, *eAtkplus1);
        } else if (!fc && std::abs(timeStep - maxTimeStepSpec) < 1e-9) {
            eAtkplus1 = taylor.eAdt(t + timeStep);
        }

        // Keep the sets of the chosen trial; propagate and reduce the particular solution of U.
        Chosen ch;
        if (fc) {
            ch.boxFc = set.boxFc[kIterChosen];
            ch.boxFG = set.boxFG[kIterChosen];
            ch.GuC = set.GuC[kIterChosen];
            ch.GuG = set.GuG[kIterChosen];
        }
        if (P.isU) {
            ch.PUzero = set.PUzero[kIterChosen];
            ch.PUinf = set.PUinf[kIterChosen];
            ch.PUAsum = set.PUAsum[kIterChosen];
        }
        if (static_cast<int>(errs.step_red.size()) <= k) errs.step_red.resize(k + 1);
        errs.step_red[k] = aux_reduce(taylor, t, set, ch, errs.bound_red[k][kIterChosen], P.isU);
        errs.accumulateErrors(k, kIterChosen);
        errs.removeRedundantValues(k, kIterChosen);

        // The particular solution of the constant input is a point: it adds no error.
        const int idx = timeStepIdx[kIterChosen];
        if (P.isu) {
            if (!savedata.Pu[idx - 1])
                savedata.Pu[idx - 1] = aux_particularConstant(taylor, P.u, timeStep);
            set.Pu = *savedata.Pu[idx - 1];
            set.Putotal = eAdtk.matmul(set.Putotal) + set.Pu;
        }

        // Propagation of the sets -----------------------------------------------------------
        if (!fc) {
            // No sets in a verified step; the time-point set is needed to restart the full one.
            RcontError.push_back(kNaN);
            RcontTpError.push_back(kNaN);
            if (std::abs(timeStep - maxTimeStepSpec) < 1e-9)
                set.Hstartp = Zonotope(eAtkplus1->matmul(params.R0.c) + set.Putotal,
                                       eAtkplus1->matmul(params.R0.G));
            out.timeInt.push_back(std::nullopt);
            out.timeIntError.push_back(kNaN);
            out.timePoint.push_back(std::nullopt);
            out.timePointError.push_back(kNaN);
        } else {
            set.Hstartp = Zonotope(eAdtk.matmul(set.startset->c) + set.Pu,
                                   eAdtk.matmul(set.startset->G));
            set.enc = set.startset->linComb(*set.Hstartp);
            double Rc, Rtp;
            errs.fullErrors(k, Rc, Rtp);
            RcontError.push_back(Rc);
            RcontTpError.push_back(Rtp);

            // The output sets: homogeneous and particular solutions are mapped by C first.
            const Tensor infty = set.G_PUtotal_infty;
            std::vector<Tensor> Gout{mapC(set.enc->G)};
            std::vector<Tensor> Gtp{mapC(set.Hstartp->G)};
            Tensor boxes = ch.boxFG + ch.GuG;
            if (P.isU) {
                boxes = infty + ch.boxFG + ch.GuG;
                if (!set.G_PUtotal_zero.empty()) {
                    const Tensor Gzero = aux_fromColumns(A, set.G_PUtotal_zero, n);
                    Gout.push_back(mapC(Gzero));
                    Gtp.push_back(mapC(Gzero));
                }
                Gtp.push_back(mapC(infty.diag()));
            }
            Gout.push_back(mapC(boxes.diag()));
            out.timeInt.push_back(Zonotope(mapC(set.enc->c + ch.boxFc + ch.GuC),
                                           Tensor::catLast(Gout)));
            out.timeIntError.push_back(errR2Y * Rc);
            out.timePoint.push_back(Zonotope(mapC(set.Hstartp->c), Tensor::catLast(Gtp)));
            out.timePointError.push_back(errR2Y * Rtp);
        }

        // The next step starts where this one ends; the approximation functions are now usable.
        t += timeStep;
        out.time.push_back(t);
        errs.useApproxFun = true;
    }

    // The errors respect their bounds (a failure is a bug).
    if (!errs.checkErrors(RcontError, RcontTpError, out.internalChecksOk))
        throw std::logic_error("priv_reach_adaptive: the error bounds were not respected");
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
