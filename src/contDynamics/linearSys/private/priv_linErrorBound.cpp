// priv_linErrorBound - the error bookkeeping of the adaptive reachability algorithm
//
// As CORA's linErrorBound.
// The methods of LinErrorBound: bounds per step, accumulation of the committed errors, the
// consistency check at the end, the allocation of emax to the reduction error, and the
// prediction of the next time step size by approximation functions or bisection.
//
// Syntax:   LinErrorBound errs(emax, tFinal);   errs.nextBounds(timeStep, t, k, kIter);
// Inputs:   emax - error margin;  tFinal - time horizon
// Outputs:  the bounds and checks of priv_linErrorBound.h
// See also: priv_reach_adaptive

#include "contDynamics/linearSys/private/priv_linErrorBound.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kEps = 2.220446049250313e-16;  // as MATLAB's eps
constexpr double kNaN =
    std::numeric_limits<double>::quiet_NaN();

/// Sets v[k][it], growing the lists as MATLAB does when an element is assigned.
template <class T>
void aux_put(std::vector<std::vector<T>> &v, int k, int it, T val) {
    if (static_cast<int>(v.size()) <= k) v.resize(k + 1);
    if (static_cast<int>(v[k].size()) <= it) v[k].resize(it + 1, T());
    v[k][it] = val;
}

/// The smallest and largest of the values that are not NaN (MATLAB's min and max).
double aux_min(std::initializer_list<double> values) {
    double m = std::numeric_limits<double>::infinity();
    bool any = false;
    for (double v : values)
        if (!std::isnan(v)) m = any ? std::min(m, v) : v, any = true;
    return any ? m : kNaN;
}

double aux_max(std::initializer_list<double> values) {
    double m = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (double v : values)
        if (!std::isnan(v)) m = any ? std::max(m, v) : v, any = true;
    return any ? m : kNaN;
}

/// Whether any entry of the list is below (or above) the limit.
bool aux_anyBelow(const std::vector<double> &v, double limit) {
    for (double x : v)
        if (x < limit) return true;
    return false;
}

bool aux_anyAbove(const std::vector<double> &v, double limit) {
    for (double x : v)
        if (x > limit) return true;
    return false;
}

/// The condition number (2-norm) of the matrix [a b; c d], from its singular values.
double aux_cond2x2(double a, double b, double c, double d) {
    const double frob2 = a * a + b * b + c * c + d * d;
    const double det = std::abs(a * d - b * c);
    const double plus = std::sqrt(std::max(frob2 + 2 * det, 0.0));
    const double minus = std::sqrt(std::max(frob2 - 2 * det, 0.0));
    return (plus + minus) / (plus - minus);
}

/// The solution of [a b; c d] x = r by LU with partial pivoting, as MATLAB's backslash.
void aux_solve2x2(
    double a, double b, double c, double d, double r1, double r2, double &x1, double &x2) {
    if (std::abs(c) > std::abs(a)) {
        std::swap(a, c);
        std::swap(b, d);
        std::swap(r1, r2);
    }
    const double l = c / a;
    const double u22 = d - l * b;
    const double y2 = r2 - l * r1;
    x2 = y2 / u22;
    x1 = (r1 - b * x2) / a;
}

/// The 2-norm of the column sums |V| of a matrix (n, m): MATLAB's vecnorm(sum(abs(V), 2)).
double aux_rowSumNorm(const Tensor &V) {
    double s = 0;
    for (double v : V.abs().sumLast().data()) s += v * v;
    return std::sqrt(s);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Bounds and bookkeeping --------------------------------------------------------------------------

LinErrorBound::LinErrorBound(double emax_, double tFinal_) : emax(emax_), tFinal(tFinal_) {
    if (!std::isfinite(emax_) || emax_ <= 0)
        throw std::invalid_argument("linErrorBound: the error margin must be a positive number");
    if (!std::isfinite(tFinal_) || tFinal_ < 0)
        throw std::invalid_argument("linErrorBound: the time horizon must be non-negative");
}

void LinErrorBound::nextBounds(double timeStep, double t, int k, int kIter) {
    aux_put(timeSteps, k, kIter, timeStep);
    // The margin of the step: emax without the final reduction error and the error accumulated.
    if (static_cast<int>(bound_remacc.size()) <= k) bound_remacc.resize(k + 1);
    bound_remacc[k] = k == 0 ? emax - bound_red_max : emax - bound_red_max - cum_acc[k - 1];
    // The accumulating error may use the share of the margin that the step has of the time left.
    aux_put(bound_acc, k, kIter, bound_remacc[k] * timeStep / (tFinal - t));
    const double boundRedTk = bound_red_max * (t + timeStep) / tFinal;
    if (k == 0) {
        aux_put(bound_red, k, kIter, boundRedTk);
        aux_put(bound_rem, k, kIter, emax - boundRedTk);
    } else {
        aux_put(bound_red, k, kIter, boundRedTk - cum_red[k - 1]);
        aux_put(bound_rem, k, kIter, emax - boundRedTk - cum_acc[k - 1]);
    }
}

void LinErrorBound::accumulateErrors(int k, int kIter) {
    if (k == 0) {
        cum_acc = {step_acc[k][kIter]};
        cum_red = {step_red[k]};
    } else {
        cum_acc.push_back(cum_acc.back() + step_acc[k][kIter]);
        cum_red.push_back(cum_red.back() + step_red[k]);
    }
}

void LinErrorBound::removeRedundantValues(int k, int kIter) {
    for (PerStep *v : {&seq_nonacc, &idv_PUtkplus1, &idv_F, &idv_G, &idv_linComb, &idv_PUtauk,
                       &step_acc, &bound_rem, &bound_acc, &bound_red})
        (*v)[k] = {(*v)[k][kIter]};
    // The step size is the last per-step list.
    timeSteps[k] = {timeSteps[k][kIter]};
}

void LinErrorBound::fullErrors(int k, double &RcontError, double &RcontTpError) const {
    // The non-accumulating error contains the accumulating error of the last step.
    RcontError = k == 0 ? seq_nonacc[k][0] + cum_red[k]
                        : seq_nonacc[k][0] + cum_acc[k - 1] + cum_red[k];
    RcontTpError = cum_acc[k] + cum_red[k];
}

// Checks ------------------------------------------------------------------------------------------

bool LinErrorBound::checkErrors(const std::vector<double> &RoutError,
                                const std::vector<double> &RoutTpError, bool &internalOk) const {
    bool res = true;
    internalOk = true;
    const std::size_t K = timeSteps.size();
    std::vector<double> tVec, nonacc, acc, nonaccBound, accBound, redBound;
    for (std::size_t k = 0; k < K; ++k) {
        tVec.push_back(timeSteps[k][0]);
        nonacc.push_back(seq_nonacc[k][0]);
        acc.push_back(step_acc[k][0]);
        nonaccBound.push_back(bound_rem[k][0]);
        accBound.push_back(bound_acc[k][0]);
        redBound.push_back(bound_red[k][0]);
    }
    std::vector<char> full(K);
    for (std::size_t k = 0; k < K; ++k) full[k] = !std::isnan(nonacc[k]);
    std::vector<double> tp = RoutTpError;
    if (tp.empty()) tp.assign(K, 0.0);

    // The bounds and errors the committed values are checked against.
    std::vector<double> accTotalBound(K), totalError(K), totalErrorTp(K);
    for (std::size_t k = 0; k < K; ++k) {
        accTotalBound[k] = cum_acc[k] + accBound[k] - acc[k];
        totalError[k] = nonacc[k] + cum_acc[k] - acc[k] + cum_red[k];
        totalErrorTp[k] = cum_acc[k] + cum_red[k];
    }

    // All errors and bounds are non-negative; the full errors stay below emax.
    std::vector<double> fullErr, fullTp, fullNonacc;
    for (std::size_t k = 0; k < K; ++k)
        if (full[k]) {
            fullErr.push_back(RoutError[k]);
            fullTp.push_back(tp[k]);
            fullNonacc.push_back(nonacc[k]);
        }
    for (const std::vector<double> *v : std::initializer_list<const std::vector<double> *>{
             &fullErr, &fullTp, &fullNonacc, &acc, &step_red,
             &nonaccBound, &accBound, &redBound, &accTotalBound})
        if (aux_anyBelow(*v, 0.0)) res = false;
    if (aux_anyAbove(fullErr, emax) || aux_anyAbove(fullTp, emax)) res = false;

    // The recomputed full errors match the ongoing computation.
    bool anyTp = false;
    for (double v : tp) anyTp = anyTp || v != 0;
    for (std::size_t k = 0; k < K; ++k) {
        if (!full[k]) continue;
        if (std::abs(totalError[k] - RoutError[k]) > 1e-9) internalOk = false;
        if (anyTp && std::abs(totalErrorTp[k] - tp[k]) > 1e-9) internalOk = false;
    }

    // Non-accumulating errors, accumulating errors and their linearly increasing bound.
    if (aux_anyAbove(nonacc, emax)) res = false;
    for (std::size_t k = 0; k < K; ++k) if (nonacc[k] > nonaccBound[k]) internalOk = false;
    if (aux_anyAbove(cum_acc, emax) || aux_anyAbove(cum_acc, emax - bound_red_max)) res = false;
    for (std::size_t k = 0; k < K; ++k)
        if (acc[k] > cum_acc[k] || acc[k] > accBound[k]) internalOk = false;
    const double finalValue = emax - bound_red_max;
    double cumT = 0;
    std::vector<double> cumsumT(K);
    for (std::size_t k = 0; k < K; ++k) cumsumT[k] = cumT += tVec[k];
    for (std::size_t k = 0; k < K; ++k) {
        const double linBound = finalValue * cumsumT[k] / cumsumT[K - 1];
        if (cum_acc[k] > linBound || linBound - accTotalBound[k] < -10 * kEps) internalOk = false;
    }

    // Reduction errors.
    if (aux_anyAbove(cum_red, emax)) res = false;
    for (std::size_t k = 0; k < K; ++k)
        if (step_red[k] > cum_red[k] || step_red[k] > redBound[k] || cum_red[k] > bound_red_max)
            internalOk = false;
    return res;
}

// Reduction error ---------------------------------------------------------------------------------

void LinErrorBound::computeErrorBoundReduction(const Tensor &A, const Tensor &GU) {
    // No input generators: nothing to reduce.
    bool any = false;
    for (double v : GU.data()) any = any || v != 0;
    if (!any) {
        bound_red_max = 0;
        return;
    }
    const int stepsForErrorBound = 100;
    const double timeStep = tFinal / stepsForErrorBound;
    Tensor eAt = A.eyeLike();
    const Tensor eADt = (A * timeStep).expm();

    // The errors of the auxiliary sets V_k = e^{A k dt} dt GU.
    const Tensor DtU = GU * timeStep;
    std::vector<double> errV(stepsForErrorBound);
    for (int i = 0; i < stepsForErrorBound; ++i) {
        const Tensor V = eAt.matmul(DtU);
        eAt = eAt.matmul(eADt);
        errV[i] = aux_rowSumNorm(V);
    }

    // Weights, ordering and the cumulative error of the ordered sets.
    double total = 0;
    for (double e : errV) total += e;
    std::vector<int> tau(stepsForErrorBound);
    std::iota(tau.begin(), tau.end(), 0);
    std::stable_sort(tau.begin(), tau.end(),
                     [&](int a, int b) { return errV[a] / total < errV[b] / total; });
    std::vector<double> cumsum(stepsForErrorBound);
    double run = 0;
    for (int i = 0; i < stepsForErrorBound; ++i) cumsum[i] = run += errV[tau[i]];

    // How much could be reduced if a share of emax were allocated to the reduction error?
    const int meshsize = 1000;
    const double d = 1.0 / (meshsize - 1);
    double bestFinalOrder = stepsForErrorBound + 1;
    int minIdx = 1;
    for (int i = 1; i <= meshsize - 1; ++i) {
        const double pct = (i - 1) * d;
        int idx = 0;
        for (int j = stepsForErrorBound; j >= 1; --j)
            if (cumsum[j - 1] < emax * pct) {
                idx = j;
                break;
            }
        const double N = 1.0 / (1.0 - pct);
        const double finalOrder = idx == 0 ? N * (stepsForErrorBound + 1)
                                           : N * (stepsForErrorBound + 1 - idx);
        if (finalOrder < bestFinalOrder) {
            bestFinalOrder = finalOrder;
            // CORA takes the index into the mesh from the count of reducible sets.
            minIdx = idx;
        }
    }
    bound_red_max = emax * ((minIdx - 1) * d);
}

// Approximation functions -------------------------------------------------------------------------

void LinErrorBound::updateCoefficientsApproxFun(int k, int kIter, bool fullcomp) {
    if (!useApproxFun) return;
    const double dt = timeSteps[k][kIter];
    if (kIter == 0) {
        // Initial guess: b = 0 for the quadratic functions.
        if (fullcomp) {
            coeff_linComb_b = idv_linComb[k][kIter] / dt;
            coeff_PUtauk_b = idv_PUtauk[k][kIter] / dt;
            coeff_F_a = idv_F[k][kIter] / (dt * dt);
            coeff_F_b = 0;
            coeff_G_a = idv_G[k][kIter] / (dt * dt);
            coeff_G_b = 0;
        }
        coeff_PUtkplus1_a = idv_PUtkplus1[k][kIter] / (dt * dt);
        coeff_PUtkplus1_b = 0;
        return;
    }
    if (fullcomp) {
        coeff_linComb_b = idv_linComb[k][kIter] / dt;
        coeff_PUtauk_b = idv_PUtauk[k][kIter] / dt;
    }
    // a dt^2 + b dt through the two most recent values.
    const double dtPrev = timeSteps[k][kIter - 1];
    const double m11 = dtPrev * dtPrev, m12 = dtPrev, m21 = dt * dt, m22 = dt;
    if (std::abs(1 / aux_cond2x2(m11, m12, m21, m22)) < kEps)
        throw std::runtime_error("LinearSys::verify: estimation of the time step size failed");
    aux_solve2x2(m11, m12, m21, m22, idv_PUtkplus1[k][kIter - 1], idv_PUtkplus1[k][kIter],
                 coeff_PUtkplus1_a, coeff_PUtkplus1_b);
    if (fullcomp) {
        aux_solve2x2(m11, m12, m21, m22, idv_F[k][kIter - 1], idv_F[k][kIter], coeff_F_a,
                     coeff_F_b);
        if (coeff_F_b < 0) {
            // A negative b cannot occur (the error is positive): use the last value only.
            coeff_F_b = 0;
            coeff_F_a = idv_F[k][kIter] / (dt * dt);
        }
        aux_solve2x2(m11, m12, m21, m22, idv_G[k][kIter - 1], idv_G[k][kIter], coeff_G_a,
                     coeff_G_b);
    }
}

// Bisection ---------------------------------------------------------------------------------------

void LinErrorBound::updateBisection(int k, int kIter, bool isU, double timeStep) {
    const bool accOk = bound_acc_ok[k][kIter], nonaccOk = bound_nonacc_ok[k][kIter];
    const double accPerc = step_acc[k][kIter] / bound_acc[k][kIter];
    const double nonaccPerc = seq_nonacc[k][kIter] / bound_rem[k][kIter];
    if (kIter == 0) {
        // The lower bound is 0; the upper bound is the first step size.
        bisect_lb_timeStep_acc = bisect_lb_acc = bisect_lb_acc_perc = 0;
        bisect_lb_accok = true;
        bisect_lb_timeStep_nonacc = bisect_lb_nonacc = bisect_lb_nonacc_perc = 0;
        bisect_lb_nonaccok = true;
        bisect_ub_timeStep_acc = timeStep;
        bisect_ub_acc = idv_PUtkplus1[k][kIter];
        bisect_ub_accok = accOk;
        bisect_ub_acc_perc = accPerc;
        bisect_ub_timeStep_nonacc = timeStep;
        bisect_ub_nonacc = seq_nonacc[k][kIter];
        bisect_ub_nonaccok = nonaccOk;
        bisect_ub_nonacc_perc = nonaccPerc;
        return;
    }
    if (isU) {
        if (!accOk || timeStep > bisect_ub_timeStep_acc) {
            // A step size that fails the check, or exceeds the bracket, is the new upper bound.
            if (timeStep > bisect_ub_timeStep_acc) {
                bisect_lb_timeStep_acc = bisect_ub_timeStep_acc;
                bisect_lb_acc = bisect_ub_acc;
                bisect_lb_accok = bisect_ub_accok;
                bisect_lb_acc_perc = bisect_ub_acc_perc;
            }
            bisect_ub_timeStep_acc = timeStep;
            bisect_ub_acc = idv_PUtkplus1[k][kIter];
            bisect_ub_accok = accOk;
            bisect_ub_acc_perc = accPerc;
        } else {
            bisect_lb_timeStep_acc = timeStep;
            bisect_lb_acc = idv_PUtkplus1[k][kIter];
            bisect_lb_accok = accOk;
            bisect_lb_acc_perc = accPerc;
        }
    }
    // The same for the non-accumulating error.
    if (!nonaccOk || timeStep > bisect_ub_timeStep_nonacc) {
        if (timeStep > bisect_ub_timeStep_nonacc) {
            bisect_lb_timeStep_nonacc = bisect_ub_timeStep_nonacc;
            bisect_lb_nonacc = bisect_ub_nonacc;
            bisect_lb_nonaccok = bisect_ub_nonaccok;
            bisect_lb_nonacc_perc = bisect_ub_nonacc_perc;
        }
        bisect_ub_timeStep_nonacc = timeStep;
        bisect_ub_nonacc = seq_nonacc[k][kIter];
        bisect_ub_nonaccok = nonaccOk;
        bisect_ub_nonacc_perc = nonaccPerc;
    } else {
        bisect_lb_timeStep_nonacc = timeStep;
        bisect_lb_nonacc = seq_nonacc[k][kIter];
        bisect_lb_nonaccok = nonaccOk;
        bisect_lb_nonacc_perc = nonaccPerc;
    }
}

// Step size prediction ----------------------------------------------------------------------------

double LinErrorBound::approxFun(double t, int k, int kIter, bool fullcomp, double timeStep,
                                double maxTimeStep) const {
    // The first guess of the first step may be far too large: shrink it before predicting.
    if (k == 0 && (seq_nonacc[k][kIter] / bound_rem[k][kIter] > 1e3 ||
                   step_acc[k][kIter] / bound_acc[k][kIter] > 1e3))
        return 0.01 * timeStep;

    // The safety factor makes the bounds more likely to hold if the functions underestimate.
    const double safetyFactor = 0.90;

    // 1. The accumulating error meets the linearly increasing bound up to the time horizon.
    const double predLinBound =
        safetyFactor *
        (1 / coeff_PUtkplus1_a * (bound_remacc[k] / (tFinal - t) - coeff_PUtkplus1_b));

    // 2. All errors meet the remaining error of the step: a quadratic equation.
    double predErem = std::numeric_limits<double>::infinity();
    if (fullcomp) {
        const double a = coeff_PUtkplus1_a + coeff_F_a + coeff_G_a;
        const double b = coeff_PUtkplus1_b + coeff_F_b + coeff_G_b + coeff_linComb_b +
                         coeff_PUtauk_b;
        const double c = -bound_rem[k][kIter];
        const double root = std::sqrt(b * b - 4 * a * c);
        predErem = safetyFactor * aux_max({1 / (2 * a) * (-b + root), 1 / (2 * a) * (-b - root)});
    }
    return aux_min({maxTimeStep, predLinBound, predErem});
}

double LinErrorBound::bisection(double t, int k, bool fullcomp, double timeStep,
                                double maxTimeStep, bool isU) const {
    (void)timeStep;
    const double eaccTotal = k == 0 ? 0.0 : cum_acc[k - 1];
    const double inf = std::numeric_limits<double>::infinity();

    // 1. Accumulating error.
    double predEacc = inf, slopeAcc = 0;
    if (isU) {
        if (bisect_ub_accok) {
            // Extrapolate the linear trend of the error up to the linear bound.
            const double boundSlope = (emax - bound_red_max) / tFinal;
            slopeAcc = (bisect_ub_acc - bisect_lb_acc) /
                       (bisect_ub_timeStep_acc - bisect_lb_timeStep_acc);
            if (slopeAcc < boundSlope) {
                predEacc = inf;
            } else {
                const double add = (boundSlope * (t + bisect_lb_timeStep_acc) - eaccTotal -
                                    bisect_lb_acc) / (slopeAcc - boundSlope);
                predEacc = bisect_lb_timeStep_acc + add;
            }
        } else {
            // Bisection: the first step halves; later ones assume a quadratic decrease.
            const double factor =
                k == 0 ? 0.5
                       : std::sqrt((0.95 - bisect_lb_acc_perc) /
                                   (bisect_ub_acc_perc - bisect_lb_acc_perc));
            predEacc = bisect_lb_timeStep_acc +
                       factor * (bisect_ub_timeStep_acc - bisect_lb_timeStep_acc);
        }
    }

    // 2. Non-accumulating error.
    double predEnonacc = inf;
    if (fullcomp) {
        if (bisect_ub_nonaccok) {
            const double slopeNonacc = (bisect_ub_nonacc - bisect_lb_nonacc) /
                                       (bisect_ub_timeStep_nonacc - bisect_lb_timeStep_nonacc);
            double add;
            if (isU)
                add = (emax - bound_red_max * t / tFinal - eaccTotal - bisect_lb_acc -
                       bisect_lb_nonacc) / (slopeAcc + slopeNonacc - bound_red_max / tFinal);
            else
                add = (emax - bisect_lb_nonacc) / slopeNonacc;
            predEnonacc = bisect_lb_timeStep_nonacc + add;
        } else {
            const double factor =
                k == 0 ? 0.5
                       : (0.95 - bisect_lb_nonacc_perc) /
                             (bisect_ub_nonacc_perc - bisect_lb_nonacc_perc);
            predEnonacc = bisect_lb_timeStep_nonacc +
                          factor * (bisect_ub_timeStep_nonacc - bisect_lb_timeStep_nonacc);
        }
        // For safety if the slopes misbehave.
        if (predEnonacc < 0) predEnonacc = inf;
    }
    return aux_min({predEnonacc, predEacc, maxTimeStep});
}

double LinErrorBound::estimateTimeStepSize(double t, int k, int kIter, bool fullcomp,
                                           double timeStep, double maxTimeStep, bool isU) {
    if (useApproxFun) {
        timeStep = approxFun(t, k, kIter, fullcomp, timeStep, maxTimeStep);
        // The proposal is refused if it lies below a lower bracket, above an upper bracket that
        // failed, or below an upper bracket that held.
        const double lbs[2] = {bisect_lb_timeStep_acc, bisect_lb_timeStep_nonacc};
        const double ubs[2] = {bisect_ub_timeStep_acc, bisect_ub_timeStep_nonacc};
        const bool oks[2] = {bisect_ub_accok, bisect_ub_nonaccok};
        bool refuse = false;
        for (int i = 0; i < 2; ++i)
            refuse = refuse || timeStep <= lbs[i] || (timeStep >= ubs[i] && !oks[i]) ||
                     (timeStep <= ubs[i] && oks[i]);
        useApproxFun = !refuse;
    }
    // The approximation functions may have been dropped above.
    if (!useApproxFun) timeStep = bisection(t, k, fullcomp, timeStep, maxTimeStep, isU);
    return timeStep;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
