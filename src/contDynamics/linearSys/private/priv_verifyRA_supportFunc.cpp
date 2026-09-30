// priv_verifyRA_supportFunc - reach-avoid verification of a linear system with support functions
//
// Quicker than reachability: the set is never computed, only its distance to each halfspace
// C_s y <= d_s (every specification is turned into a safe set), by the support function of the
// affine solution along the back-propagated directions l_k = (e^{A dt}')^k (C_s C)'. A step size
// is tried, the distances are checked, and the step shrinks until every halfspace holds for the
// whole horizon or a trajectory of R0 and the centre of U hits one. Supported as in the ARCH
// benchmarks: x' = A x + B u, y = C x, a zonotope R0 and a constant zonotope U, no offsets.
//
// Syntax:   res = priv_verifyRA_supportFunc(sys, params, specs);
// Inputs:   sys - linear system;  params - R0, U, tFinal;  specs - safe and unsafe halfspaces
// Outputs:  res - verified (false if undecided at a step below 1e-12), iterations, last step
//           size and count, tComp, and the falsifying initial state and time
// See also: LinearSys::verify

#include "contDynamics/linearSys/private/priv_verify.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Constants and types -----------------------------------------------------------------------------

constexpr double kEps = 2.220446049250313e-16;  // DBL_EPSILON, as MATLAB's eps
constexpr int kMaxEta = 75;           // Taylor terms before a step size counts as not converged
constexpr int kStepsStart = 100;      // steps of the first pass
constexpr double kShrinkFixed = 0.2;  // step size factor of a pass that does not decide
constexpr double kShrinkNonconverged = 0.2;  // factor when a Taylor sum does not converge
constexpr double kMinTimeStep = 1e-12;

using Intervals = std::vector<std::array<double, 2>>;

/// A^i / i! and its positive and negative parts, kept over the passes; A^-1 if A is invertible.
struct Expmat {
    std::vector<Tensor> Apower, Apos, Aneg;
    std::optional<Tensor> Ainv;
};

/// An interval matrix as its center and radius.
struct IntMat {
    Tensor center, rad;
};

/// The halfspaces Cs y <= ds, row-major (nr, p), after every specification is a safe set.
struct SpecRows {
    std::vector<double> Cs, ds;
    int nr = 0;
};

/// What a pass propagates: the directions l_k as rows (k * nr + s), their products with R0's
/// generators, and the distances of the affine solution and the curvature enlargement.
struct Pass {
    Tensor LT;
    std::vector<double> ltg, affine_tp, Cbloat;
};

// Linear algebra ----------------------------------------------------------------------------------

/// The largest entry of t.
double aux_maxAll(const Tensor &t) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : t.maxLast().data()) m = std::max(m, v);
    return m;
}

/// Whether a <= b holds for every entry.
bool aux_allLE(const Tensor &a, const Tensor &b) { return aux_maxAll(a - b) <= 0; }

/// The inverse by Gauss-Jordan with partial pivoting; empty if A is (numerically) singular.
std::optional<Tensor> aux_inverse(const Tensor &A) {
    const int64_t n = A.shape()[0];
    std::vector<double> M = A.data(), inv(n * n, 0.0);
    double maxAbs = 0;
    for (double v : M) maxAbs = std::max(maxAbs, std::abs(v));
    for (int64_t i = 0; i < n; ++i) inv[i * n + i] = 1;
    // Same tolerance scale as MATLAB's rank: size * eps * norm.
    const double tol = double(n) * kEps * maxAbs;
    for (int64_t j = 0; j < n; ++j) {
        int64_t piv = j;
        for (int64_t i = j + 1; i < n; ++i)
            if (std::abs(M[i * n + j]) > std::abs(M[piv * n + j])) piv = i;
        if (std::abs(M[piv * n + j]) <= tol) return std::nullopt;
        if (piv != j) {
            std::swap_ranges(M.begin() + j * n, M.begin() + (j + 1) * n, M.begin() + piv * n);
            std::swap_ranges(inv.begin() + j * n, inv.begin() + (j + 1) * n, inv.begin() + piv * n);
        }
        const double d = M[j * n + j];
        for (int64_t c = 0; c < n; ++c) {
            M[j * n + c] /= d;
            inv[j * n + c] /= d;
        }
        for (int64_t i = 0; i < n; ++i) {
            const double f = M[i * n + j];
            if (i == j || f == 0) continue;
            for (int64_t c = j; c < n; ++c) M[i * n + c] -= f * M[j * n + c];
            for (int64_t c = 0; c < n; ++c) inv[i * n + c] -= f * inv[j * n + c];
        }
    }
    return Tensor::like(A, inv, {n, n});
}

// Taylor sums -------------------------------------------------------------------------------------

/// Makes Apower hold A^i / i! up to i = eta (entry i - 1).
void aux_getApower(Expmat &E, const Tensor &A, int eta) {
    while (static_cast<int>(E.Apower.size()) < eta) {
        const int i = static_cast<int>(E.Apower.size());
        E.Apower.push_back(E.Apower.back().matmul(A) * (1.0 / (i + 1)));
    }
}

/// Makes Apos and Aneg hold the parts of A^eta / eta! (Apower must reach eta).
void aux_getAposneg(Expmat &E, int eta) {
    if (static_cast<int>(E.Apos.size()) < eta) {
        E.Apos.resize(eta);
        E.Aneg.resize(eta);
    }
    if (!E.Apos[eta - 1].defined()) {
        E.Apos[eta - 1] = E.Apower[eta - 1].pos();
        E.Aneg[eta - 1] = E.Apower[eta - 1].neg();
    }
}

/// The interval matrices F (state) and G (input, if isu) of the curvature enlargement by Taylor
/// series until the terms no longer change the sums; false if that takes kMaxEta terms.
bool aux_intmat(const Tensor &A, bool isu, Expmat &E, double timeStep, IntMat &F, IntMat &G) {
    Tensor posF = A.zerosLike(), negF = A.zerosLike(), posG = A.zerosLike(), negG = A.zerosLike();
    bool stopF = false, stopG = !isu;
    if (!isu) G = {A.zerosLike(), A.zerosLike()};
    for (int eta = 2; !(stopF && stopG); ++eta) {
        aux_getApower(E, A, eta);
        const double factor = (std::pow(eta, -double(eta) / (eta - 1)) -
                               std::pow(eta, -1.0 / (eta - 1))) * std::pow(timeStep, eta);
        if (!stopF) {
            aux_getAposneg(E, eta);
            const Tensor addPos = E.Aneg[eta - 1] * factor, addNeg = E.Apos[eta - 1] * factor;
            if (eta == kMaxEta) return false;
            if (aux_allLE(addPos, posF * kEps) && aux_allLE(negF * kEps, addNeg)) {
                stopF = true;
                F.rad = (posF - negF) * 0.5;
                F.center = negF + F.rad;
            }
            posF = posF + addPos;
            negF = negF + addNeg;
        }
        if (!stopG) {
            // The stored terms carry 1/(eta-1)!, so G needs one more division by eta.
            aux_getAposneg(E, eta - 1);
            const Tensor addPos = E.Aneg[eta - 2] * (factor / eta);
            const Tensor addNeg = E.Apos[eta - 2] * (factor / eta);
            if (eta == kMaxEta) return false;
            if (aux_allLE(addPos, posG * kEps) && aux_allLE(negG * kEps, addNeg)) {
                stopG = true;
                G.rad = (posG - negG) * 0.5;
                G.center = negG + G.rad;
            }
            posG = posG + addPos;
            negG = negG + addNeg;
        }
    }
    return true;
}

/// The matrix X of a particular solution A^-1 (e^{A dt} - I) X, by the series
/// sum_j A^{j-1} dt^j / j! until it stops changing if A is singular; empty if it diverges.
std::optional<Tensor> aux_particular(
    const Tensor &A, const Tensor &X, Expmat &E, const Tensor &Delta, double timeStep) {
    if (E.Ainv) return E.Ainv->matmul((Delta - A.eyeLike()).matmul(X));
    Tensor Asum = A.eyeLike() * timeStep;
    for (int eta = 2;; ++eta) {
        aux_getApower(E, A, eta - 1);
        const Tensor addTerm = E.Apower[eta - 2] * (std::pow(timeStep, eta) / eta);
        // Too large a step size diverges; an unbounded loop would never end.
        if (std::isinf(aux_maxAll(addTerm.abs())) || eta > 1000) return std::nullopt;
        if (aux_allLE(addTerm.abs(), Asum.abs() * kEps)) break;
        Asum = Asum + addTerm;
    }
    return Asum.matmul(X);
}

/// The generators of the outer approximation of the particular solution of U over one step,
/// [dt G, A dt^2/2 G, ...]; empty if the series needs kMaxEta terms.
std::optional<Tensor> aux_overPU(const Tensor &A, const Tensor &GU, Expmat &E, double timeStep) {
    std::vector<Tensor> parts{GU * timeStep};
    std::vector<double> diag = parts[0].abs().sumLast().data();
    for (int eta = 1;;) {
        aux_getApower(E, A, eta);
        const Tensor add = E.Apower[eta - 1].matmul(GU) * (std::pow(timeStep, eta + 1) / (eta + 1));
        const std::vector<double> addDiag = add.abs().sumLast().data();
        bool stop = true;
        for (std::size_t i = 0; i < diag.size(); ++i) {
            stop = stop && addDiag[i] <= kEps * diag[i];
            diag[i] += addDiag[i];
        }
        parts.push_back(add);
        if (stop) break;
        if (++eta == kMaxEta) return std::nullopt;
    }
    return Tensor::catLast(parts);
}

// Specifications ----------------------------------------------------------------------------------

/// The halfspaces of the specifications as safe sets C y <= d with unit normals: an unsafe
/// set a'y <= b is the safe set -a'y <= -b.
SpecRows aux_specRows(const std::vector<Specification> &specs, int64_t p) {
    if (specs.empty()) throw std::invalid_argument("LinearSys::verify: no specification given");
    SpecRows rows;
    // A halfspace with a unit normal: scaling does not change which points satisfy it.
    auto add = [&](const Halfspace &h, double sign) {
        const std::vector<int64_t> shape = h.a.shape();
        if (shape.size() != 2 || shape[0] != p || shape[1] != 1)
            throw std::invalid_argument(
                "LinearSys::verify: a halfspace normal must be a column of the output dimension");
        const std::vector<double> a = h.a.data();
        double norm = 0;
        for (double v : a) norm += v * v;
        norm = std::sqrt(norm);
        if (!(norm > 0))
            throw std::invalid_argument("LinearSys::verify: a halfspace normal must not be zero");
        for (double v : a) rows.Cs.push_back(sign * v / norm);
        rows.ds.push_back(sign * h.b / norm);
        ++rows.nr;
    };
    for (const Specification &spec : specs) {
        // Each halfspace of a safe set is a row of its own: the set holds if all of them do.
        switch (spec.type()) {
        case SpecType::SafeSet:
            for (const Halfspace &h : spec.halfspaces()) add(h, 1.0);
            break;
        case SpecType::UnsafeSet:
            if (spec.halfspaces().size() != 1)
                throw std::invalid_argument(
                    "LinearSys::verify: an unsafe set must be a single halfspace");
            add(spec.halfspaces()[0], -1.0);
            break;
        default:
            throw std::invalid_argument("LinearSys::verify: unknown specification type");
        }
    }
    return rows;
}

/// The largest step size <= timeStep0 that hits tFinal with a whole number of steps.
double aux_timeStep(double timeStep0, double tFinal) {
    return tFinal / std::ceil(tFinal / timeStep0);
}

/// Removes the times [t0, t1] from the unverified intervals, dropping slivers.
void aux_removeFromUnsat(Intervals &unsat, double t0, double t1) {
    Intervals kept;
    for (const std::array<double, 2> &iv : unsat) {
        if (t1 <= iv[0] || t0 >= iv[1]) {
            kept.push_back(iv);
            continue;
        }
        if (iv[0] < t0) kept.push_back({iv[0], t0});
        if (t1 < iv[1]) kept.push_back({t1, iv[1]});
    }
    kept.erase(std::remove_if(kept.begin(), kept.end(),
                              [](const std::array<double, 2> &iv) {
                                  return std::abs(iv[1] - iv[0]) < 1e-14;
                              }),
               kept.end());
    unsat = std::move(kept);
}

/// The initial state c + G sign(l'G) of R0 that reaches the support value of row `row` of ltg.
Tensor aux_extremeX0(const Zonotope &R0, const std::vector<double> &ltg, std::size_t row) {
    const int64_t m0 = R0.G.shape()[1];
    std::vector<double> sgn(m0);
    for (int64_t j = 0; j < m0; ++j) {
        const double v = ltg[row * m0 + j];
        sgn[j] = (v > 0) - (v < 0);
    }
    return R0.c + R0.G.matmul(Tensor::like(R0.c, sgn, {m0, 1}));
}

// Propagation -------------------------------------------------------------------------------------

/// Propagates the directions for N steps and gathers the distances of the affine solution
/// (time points) and of the curvature enlargement (time intervals) to every halfspace.
Pass aux_pass(
    const Zonotope &R0, const Tensor &lT, const std::vector<double> &ds, bool isu,
    const Tensor &Delta, const Tensor &Pu, const Tensor &u, const IntMat &F, const IntMat &G,
    int N) {
    const int S = static_cast<int>(ds.size());
    const int64_t m0 = R0.G.shape()[1];
    Pass pass;
    std::vector<Tensor> rows{lT};
    for (int k = 0; k < N; ++k) rows.push_back(rows.back().matmul(Delta));
    pass.LT = Tensor::catRows(rows);
    const std::vector<double> ltc = pass.LT.matmul(R0.c).data();
    pass.ltg = pass.LT.matmul(R0.G).data();

    // The accumulated particular solution of the constant input, projected on l.
    std::vector<double> PuProj;
    if (isu && N > 0) {
        std::vector<Tensor> cols;
        Tensor P = Pu.zerosLike();
        for (int k = 0; k < N; ++k) cols.push_back(P = Delta.matmul(P) + Pu);
        PuProj = lT.matmul(Tensor::catLast(cols)).data();
    }
    pass.affine_tp.assign(std::size_t(N + 1) * S, 0.0);
    for (int k = 0; k <= N; ++k)
        for (int s = 0; s < S; ++s) {
            const std::size_t r = std::size_t(k) * S + s;
            double sumAbs = 0;
            for (int64_t j = 0; j < m0; ++j) sumAbs += std::abs(pass.ltg[r * m0 + j]);
            const double Pus = (isu && k > 0) ? PuProj[std::size_t(s) * N + (k - 1)] : 0.0;
            pass.affine_tp[r] = -ds[s] + ltc[r] + sumAbs + Pus;
        }

    // The curvature term F R0 + G u, evaluated along l_k for k < N.
    const Tensor cx = F.center.matmul(R0.c), G1x = F.center.matmul(R0.G);
    const Tensor G2x = F.rad.matmul(Tensor::catLast({R0.c, R0.G}).abs().sumLast());
    std::vector<Tensor> W{cx, G2x};
    if (isu) {
        W.push_back(G.center.matmul(u));
        W.push_back(G.rad.matmul(u.abs()));
    }
    const int64_t nw = W.size();
    const std::vector<double> lw = pass.LT.matmul(Tensor::catLast(W)).data();
    const std::vector<double> lg1 = pass.LT.matmul(G1x).data();
    pass.Cbloat.assign(std::size_t(N) * S, 0.0);
    for (std::size_t r = 0; r < pass.Cbloat.size(); ++r) {
        double sumAbs = 0;
        for (int64_t j = 0; j < m0; ++j) sumAbs += std::abs(lg1[r * m0 + j]);
        double c = lw[r * nw] + sumAbs + std::abs(lw[r * nw + 1]);
        if (isu) c = (c + lw[r * nw + 2]) + std::abs(lw[r * nw + 3]);
        pass.Cbloat[r] = c;
    }
    return pass;
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult priv_verifyRA_supportFunc(const LinearSys &sys, const VerifyParams &params,
                                       const std::vector<Specification> &specs) {
    const auto start = std::chrono::steady_clock::now();
    VerifyResult res;
    auto finish = [&]() {
        res.tComp = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return res;
    };

    // Validation ----------------------------------------------------------------------------------
    const Tensor &A = sys.A();
    const std::vector<int64_t> shapeA = A.shape();
    if (shapeA.size() != 2 || shapeA[0] != shapeA[1])
        throw std::invalid_argument("LinearSys::verify: A must be a single square matrix");
    const int64_t n = shapeA[0];
    const Zonotope &R0 = params.R0;
    if (R0.c.shape() != std::vector<int64_t>{n, 1} || R0.G.shape().size() != 2 ||
        R0.G.shape()[0] != n)
        throw std::invalid_argument("LinearSys::verify: R0 must be a single zonotope of dim(A)");
    if (!(params.tFinal > 0) || std::isinf(params.tFinal))
        throw std::invalid_argument("LinearSys::verify: tFinal must be positive and finite");
    if (sys.C() && (sys.C()->shape().size() != 2 || sys.C()->shape()[1] != n))
        throw std::invalid_argument("LinearSys::verify: C must be a matrix with dim(A) columns");
    const int64_t p = sys.C() ? sys.C()->shape()[0] : n;

    // The input: x' = A x + u with u in B U, split into the constant uTrans = B center(U) and
    // the uncertain part GU = B generators(U).
    Tensor u = Tensor::like(A, std::vector<double>(n, 0.0), {n, 1});
    Tensor GU = Tensor::like(A, {}, {n, 0});
    if (sys.B()) {
        const Tensor &B = *sys.B();
        if (B.shape().size() != 2 || B.shape()[0] != n ||
            params.U.c.shape() != std::vector<int64_t>{B.shape()[1], 1} ||
            params.U.G.shape().size() != 2 || params.U.G.shape()[0] != B.shape()[1])
            throw std::invalid_argument("LinearSys::verify: U must be a zonotope of dim(B.cols)");
        u = B.matmul(params.U.c);
        if (params.U.G.shape()[1] > 0) GU = B.matmul(params.U.G);
        // Zero generators add no uncertainty; dropping them keeps the no-input path.
        bool anyGen = false;
        for (double v : GU.data()) anyGen = anyGen || v != 0;
        if (!anyGen) GU = Tensor::like(A, {}, {n, 0});
    } else {
        for (double v : params.U.c.data())
            if (v != 0) throw std::invalid_argument("LinearSys::verify: U given without a B");
        for (double v : params.U.G.data())
            if (v != 0) throw std::invalid_argument("LinearSys::verify: U given without a B");
    }
    const bool isU = GU.shape()[1] > 0;
    bool isu = false;
    for (double v : u.data()) isu = isu || v != 0;

    // The directions l = (Cs C)' of the support function evaluations, per halfspace.
    const SpecRows rows = aux_specRows(specs, p);
    const int S = rows.nr;
    Tensor CsT = Tensor::like(A, rows.Cs, {S, p});
    const Tensor lT = sys.C() ? CsT.matmul(*sys.C()) : CsT;  // (S, n)
    const std::vector<double> &ds = rows.ds;

    Expmat E;
    E.Apower.push_back(A);
    if (isu || isU) E.Ainv = aux_inverse(A);

    // Initial set ---------------------------------------------------------------------------------
    // Distance of the initial output set to each halfspace; > 0 means it is already violated.
    {
        const std::vector<double> lc = lT.matmul(R0.c).data();
        const std::vector<double> lg = lT.matmul(R0.G).data();
        const int64_t m0 = R0.G.shape()[1];
        for (int s = 0; s < S; ++s) {
            double sumAbs = 0;
            for (int64_t j = 0; j < m0; ++j) sumAbs += std::abs(lg[s * m0 + j]);
            if (-ds[s] + lc[s] + sumAbs > 0) {
                res.fals = Falsification{aux_extremeX0(R0, lg, s), 0.0};
                return finish();
            }
        }
    }

    // Adaptive loop -------------------------------------------------------------------------------
    double tFinal = params.tFinal;
    double timeStep = aux_timeStep(tFinal / kStepsStart, tFinal);
    std::vector<Intervals> unsat(S, Intervals{{0.0, tFinal}});
    while (true) {
        ++res.iterations;
        const int N = static_cast<int>(std::round(tFinal / timeStep));
        res.timeStep = timeStep;
        res.nrSteps = N;
        const Tensor Delta = (A * timeStep).expm();

        // The constant input solution and the interval matrices; a slow sum shrinks the step.
        Tensor Pu = u.zerosLike();
        if (isu) {
            const std::optional<Tensor> P = aux_particular(A, u, E, Delta, timeStep);
            if (!P) {
                timeStep *= kShrinkNonconverged;
                continue;
            }
            Pu = *P;
        }
        IntMat F, G;
        if (!aux_intmat(A, isu, E, timeStep, F, G)) {
            timeStep *= kShrinkNonconverged;
            continue;
        }

        // The affine solution at the time points: a positive distance is a hit by the
        // trajectory from the extreme point of R0 with the centre of U.
        const Pass pass = aux_pass(R0, lT, ds, isu, Delta, Pu, u, F, G, N);
        for (int k = 1; k <= N; ++k)
            for (int s = 0; s < S; ++s)
                if (pass.affine_tp[std::size_t(k) * S + s] > 0) {
                    res.fals = Falsification{aux_extremeX0(R0, pass.ltg, std::size_t(k) * S + s),
                                             timeStep * k};
                    return finish();
                }

        // Outer distance over each step: the larger end point plus curvature and, with an
        // uncertain input, the outer approximation of its particular solution.
        std::vector<double> overPU(std::size_t(N + 1) * S, 0.0);
        if (isU) {
            const std::optional<Tensor> underG = aux_particular(A, GU, E, Delta, timeStep);
            if (!underG) {
                timeStep *= kShrinkNonconverged;
                continue;
            }
            const std::optional<Tensor> overG = aux_overPU(A, GU, E, timeStep);
            if (!overG) {
                timeStep *= kShrinkNonconverged;
                continue;
            }
            // Both sums accumulate over the steps; index k is the solution at t_k.
            std::vector<double> underPU(std::size_t(N + 1) * S, 0.0);
            const std::vector<double> dUnder = pass.LT.matmul(*underG).abs().sumLast().data();
            const std::vector<double> dOver = pass.LT.matmul(*overG).abs().sumLast().data();
            for (int k = 1; k <= N; ++k)
                for (int s = 0; s < S; ++s) {
                    const std::size_t r = std::size_t(k) * S + s, prev = r - S;
                    underPU[r] = underPU[prev] + dUnder[prev];
                    overPU[r] = overPU[prev] + dOver[prev];
                }
            // The inner approximation of the solution at the time points reaches a halfspace.
            for (int k = 0; k <= N; ++k)
                for (int s = 0; s < S; ++s) {
                    const std::size_t r = std::size_t(k) * S + s;
                    if (pass.affine_tp[r] + underPU[r] > 0) {
                        res.fals = Falsification{aux_extremeX0(R0, pass.ltg, r), timeStep * k};
                        return finish();
                    }
                }
        }

        // Time intervals whose outer distance is below 0 for every halfspace are verified.
        std::vector<std::vector<double>> worst(S, std::vector<double>(N));
        bool allNeg = true;
        for (int s = 0; s < S; ++s) {
            for (int k = 0; k < N; ++k) {
                const std::size_t r = std::size_t(k) * S + s, next = r + S;
                const double a = pass.affine_tp[r] + pass.Cbloat[r];
                const double b = pass.affine_tp[next] + pass.Cbloat[r];
                worst[s][k] = std::max(a, b) + overPU[next];
                allNeg = allNeg && worst[s][k] < 0;
            }
        }
        for (int s = 0; s < S; ++s) {
            bool specNeg = true;
            for (double w : worst[s]) specNeg = specNeg && w < 0;
            // Without an uncertain input the verified set is cleared only if all are verified.
            if (isU ? specNeg : allNeg) {
                unsat[s].clear();
                continue;
            }
            double t0 = 0, t1 = timeStep;
            for (int k = 0; k < N; ++k) {
                if (worst[s][k] < 0) aux_removeFromUnsat(unsat[s], t0, t1);
                t0 += timeStep;
                t1 += timeStep;
            }
        }

        // Without an uncertain input, the horizon shrinks to the last unverified time.
        bool allVerified = true;
        double lastUnsat = 0;
        for (const Intervals &iv : unsat)
            if (!iv.empty()) {
                allVerified = false;
                lastUnsat = std::max(lastUnsat, iv.back()[1]);
            }
        if (allVerified) break;
        if (!isU) tFinal = lastUnsat;
        timeStep *= kShrinkFixed;
        if (timeStep < kMinTimeStep) return finish();
    }
    res.verified = true;
    return finish();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
