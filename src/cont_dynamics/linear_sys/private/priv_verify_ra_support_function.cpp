// priv_verify_ra_support_function - reach-avoid verification of a linear system by supports
//
// Quicker than reachability: the set is never computed, only its distance to each halfspace
// C_s y <= d_s (every specification is turned into a safe set), by the support function of the
// affine solution along the back-propagated directions l_k = (e^{A dt}')^k (C_s C)'. A step size
// is tried, the distances are checked, and the step shrinks until every halfspace holds for the
// whole horizon or a trajectory of R0 and the centre of U hits one. Supported as in the ARCH
// benchmarks: x' = A x + B u, y = C x, a zonotope R0 and a constant zonotope U, no offsets.
//
// Syntax:   res = priv_verify_ra_support_function(sys, params, specs);
// Inputs:   sys - linear system;  params - R0, U, t_final;  specs - safe and unsafe halfspaces
// Outputs:  res - verified (false if undecided at a step below 1e-12), iterations, last step
//           size and count, computation_time, and the falsifying initial state and time
// See also: LinearSys::verify

#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cont_dynamics/linear_sys/private/priv_verify_time.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Constants and types -----------------------------------------------------------------------------

constexpr int kMaxEta = 75;          // Taylor terms before a step size counts as not converged
constexpr int kStepsStart = 100;     // steps of the first pass
constexpr double kShrinkFixed = 0.2; // step size factor of a pass that does not decide
constexpr double kShrinkNonconverged = 0.2; // factor when a Taylor sum does not converge

using Intervals = std::vector<std::array<double, 2>>;

/// Flushes denormal numbers to zero while it lives: the powers and squarings of a stiff matrix
/// underflow through the denormal range, where every product is slow, and nothing there counts.
struct FlushDenormals {
#if defined(__SSE2__) || defined(_M_X64)
    unsigned saved = _mm_getcsr();
    FlushDenormals() { _mm_setcsr(saved | 0x8040); }
    ~FlushDenormals() { _mm_setcsr(saved); }
#endif
};

/// A matrix by rows of nonzeros.
struct Csr {
    std::vector<int64_t> row_ptr, col_idx;
    std::vector<double> val;
};

/// A^i / i!, kept over the passes; A^-1 if A is invertible.
struct Expmat {
    std::vector<std::vector<double>> Apower; // row-major (n, n), on the host
    std::optional<Tensor> Ainv;
    // A by rows (CSR) when it is sparse: the powers then cost n nnz(A) instead of n^3.
    bool sparse_a = false;
    Csr a;
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
double aux_max_all(const Tensor &t) {
    double m = -std::numeric_limits<double>::infinity();
    for (double v : t.max(-1, true).data()) m = std::max(m, v);
    return m;
}

/// Whether a <= b holds for every entry.
bool aux_all_le(const Tensor &a, const Tensor &b) { return aux_max_all(a - b) <= 0; }

// Taylor sums -------------------------------------------------------------------------------------

/// The nonzeros of the row-major (n, n) matrix m by rows.
Csr aux_csr(const std::vector<double> &m, int64_t n) {
    Csr c;
    c.row_ptr.assign(1, 0);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j)
            if (m[i * n + j] != 0) {
                c.col_idx.push_back(j);
                c.val.push_back(m[i * n + j]);
            }
        c.row_ptr.push_back(static_cast<int64_t>(c.val.size()));
    }
    return c;
}

/// Keeps A in rows of nonzeros if fewer than a tenth of its entries are nonzero.
void aux_init_sparse(Expmat &E, const Tensor &A) {
    const int64_t n = A.shape()[0];
    Csr c = aux_csr(A.data(), n);
    if (c.val.size() < 0.1 * double(n) * double(n)) {
        E.sparse_a = true;
        E.a = std::move(c);
    }
}

/// x B / d for a dense x (n, n) and a sparse B; B's rows skip the zero entries of x.
std::vector<double> aux_mul_csr(const std::vector<double> &x, const Csr &B, int64_t n, double d) {
    std::vector<double> out(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < n; ++k) {
            const double xik = x[i * n + k];
            if (xik == 0) continue;
            for (int64_t q = B.row_ptr[k]; q < B.row_ptr[k + 1]; ++q)
                out[i * n + B.col_idx[q]] += xik * B.val[q];
        }
    for (double &v : out) v /= d;
    return out;
}

/// X A / d for a dense X (n, n) on the host.
std::vector<double> aux_times_a(const Expmat &E, const Tensor &A, const std::vector<double> &x,
                                double d) {
    const int64_t n = A.shape()[0];
    if (E.sparse_a) return aux_mul_csr(x, E.a, n, d);
    const Tensor X = Tensor::like(A, x, {n, n});
    return (X.matmul(A) * (1.0 / d)).data();
}

/// The multiplications that square m are cheaper as sparse rows than as a dense product.
bool aux_sparse_square_wins(const std::vector<double> &m, int64_t n) {
    std::vector<double> row_count(n, 0.0), col_count(n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            if (m[i * n + j] != 0) {
                row_count[i] += 1;
                col_count[j] += 1;
            }
    double flops = 0;
    for (int64_t k = 0; k < n; ++k) flops += col_count[k] * row_count[k];
    return flops < 0.1 * double(n) * double(n) * double(n);
}

/// e^{A time_step} by scaling and squaring with a Taylor series of the scaled A, whose terms are
/// products with sparse A; the first squarings stay sparse too. A dense A uses the backend.
Tensor aux_expm(const Tensor &A, const Expmat &E, double time_step) {
    if (!E.sparse_a) return (A * time_step).expm();
    const int64_t n = A.shape()[0];
    // The scaled norm the series takes: 2^k / k! is below 1e-18 at k = 27.
    const double theta = 2.0;
    std::vector<double> col_sum(n, 0.0);
    for (std::size_t q = 0; q < E.a.val.size(); ++q)
        col_sum[E.a.col_idx[q]] += std::abs(E.a.val[q]);
    const double norm1 = *std::max_element(col_sum.begin(), col_sum.end()) * time_step;
    const int s = norm1 > theta ? static_cast<int>(std::ceil(std::log2(norm1 / theta))) : 0;
    const double h = std::ldexp(time_step, -s);
    // The Taylor series of e^{A h}: term_k = term_{k-1} (A h) / k, summed until it is negligible.
    Csr X = E.a;
    for (double &v : X.val) v *= h;
    std::vector<double> sum(n * n, 0.0), term(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i) sum[i * n + i] = term[i * n + i] = 1.0;
    for (int k = 1; k <= 60; ++k) {
        term = aux_mul_csr(term, X, n, k);
        double max_term = 0, max_sum = 0;
        for (std::size_t i = 0; i < sum.size(); ++i) {
            sum[i] += term[i];
            max_term = std::max(max_term, std::abs(term[i]));
            max_sum = std::max(max_sum, std::abs(sum[i]));
        }
        if (max_term <= numerics::kExpmTermRatio * max_sum) break;
    }
    // Square back up to e^{A time_step}; sparse products while they are cheaper than a dense one.
    int squared = 0;
    while (squared < s && aux_sparse_square_wins(sum, n)) {
        sum = aux_mul_csr(sum, aux_csr(sum, n), n, 1.0);
        ++squared;
    }
    Tensor result = Tensor::like(A, sum, {n, n});
    for (; squared < s; ++squared) result = result.matmul(result);
    return result;
}

/// Makes Apower hold A^i / i! up to i = eta (entry i - 1).
void aux_get_apower(Expmat &E, const Tensor &A, int eta) {
    while (static_cast<int>(E.Apower.size()) < eta) {
        const int i = static_cast<int>(E.Apower.size());
        E.Apower.push_back(aux_times_a(E, A, E.Apower.back(), i + 1));
    }
}

/// A^i / i! as a tensor (Apower must reach i).
Tensor aux_power_tensor(const Expmat &E, const Tensor &A, int i) {
    const int64_t n = A.shape()[0];
    return Tensor::like(A, E.Apower[i - 1], {n, n});
}

/// The sums of one Taylor term a scaled by f (f < 0) into pos (from the negative part of a) and
/// neg (from the positive part). If the term no longer changes the sums, rad and center of the
/// interval matrix [neg, pos] before the term are written; the return value says whether it held.
bool aux_sum_step(const std::vector<double> &a, double f, std::vector<double> &pos,
                  std::vector<double> &neg, std::vector<double> &rad, std::vector<double> &center) {
    const std::size_t nn = a.size();
    bool converged = true;
    for (std::size_t i = 0; i < nn && converged; ++i)
        converged = f * std::min(a[i], 0.0) <= machine_eps() * pos[i] &&
                    f * std::max(a[i], 0.0) >= machine_eps() * neg[i];
    for (std::size_t i = 0; i < nn; ++i) {
        if (converged) {
            rad[i] = 0.5 * (pos[i] - neg[i]);
            center[i] = neg[i] + rad[i];
        }
        pos[i] += f * std::min(a[i], 0.0);
        neg[i] += f * std::max(a[i], 0.0);
    }
    return converged;
}

/// The interval matrices F (state) and G (input, if isu) of the curvature enlargement by Taylor
/// series until the terms no longer change the sums; false if that takes kMaxEta terms. The
/// sums run over host arrays in one pass per term, as the terms of a large A are all of n^2.
bool aux_intmat(const Tensor &A, bool isu, Expmat &E, double time_step, IntMat &F, IntMat &G) {
    const int64_t n = A.shape()[0];
    const std::size_t nn = std::size_t(n) * n;
    std::vector<double> pos_f(nn, 0.0), neg_f(nn, 0.0), rad_f(nn), cen_f(nn);
    std::vector<double> pos_g, neg_g, rad_g, cen_g;
    if (isu) pos_g.assign(nn, 0.0), neg_g.assign(nn, 0.0), rad_g.resize(nn), cen_g.resize(nn);
    bool stop_f = false, stop_g = !isu;
    for (int eta = 2; !(stop_f && stop_g); ++eta) {
        aux_get_apower(E, A, eta);
        const double factor =
            (std::pow(eta, -double(eta) / (eta - 1)) - std::pow(eta, -1.0 / (eta - 1))) *
            std::pow(time_step, eta);
        if (!stop_f) {
            if (eta == kMaxEta) return false;
            stop_f = aux_sum_step(E.Apower[eta - 1], factor, pos_f, neg_f, rad_f, cen_f);
        }
        if (!stop_g) {
            // The stored terms carry 1/(eta-1)!, so G needs one more division by eta.
            if (eta == kMaxEta) return false;
            stop_g = aux_sum_step(E.Apower[eta - 2], factor / eta, pos_g, neg_g, rad_g, cen_g);
        }
    }
    F = {Tensor::like(A, cen_f, {n, n}), Tensor::like(A, rad_f, {n, n})};
    G = isu ? IntMat{Tensor::like(A, cen_g, {n, n}), Tensor::like(A, rad_g, {n, n})}
            : IntMat{A.zeros_like(), A.zeros_like()};
    return true;
}

/// The matrix X of a particular solution A^-1 (e^{A dt} - I) X, by the series
/// sum_j A^{j-1} dt^j / j! until it stops changing if A is singular; empty if it diverges.
std::optional<Tensor> aux_particular(const Tensor &A, const Tensor &X, Expmat &E,
                                     const Tensor &Delta, double time_step) {
    if (E.Ainv) return E.Ainv->matmul((Delta - A.eye_like()).matmul(X));
    Tensor Asum = A.eye_like() * time_step;
    for (int eta = 2;; ++eta) {
        aux_get_apower(E, A, eta - 1);
        const Tensor add_term = aux_power_tensor(E, A, eta - 1) * (std::pow(time_step, eta) / eta);
        // Too large a step size diverges; an unbounded loop would never end.
        if (std::isinf(aux_max_all(add_term.abs())) || eta > 1000) return std::nullopt;
        if (aux_all_le(add_term.abs(), Asum.abs() * machine_eps())) break;
        Asum = Asum + add_term;
    }
    return Asum.matmul(X);
}

/// The generators of the outer approximation of the particular solution of U over one step,
/// [dt G, A dt^2/2 G, ...]; empty if the series needs kMaxEta terms.
std::optional<Tensor> aux_over_pu(const Tensor &A, const Tensor &GU, Expmat &E, double time_step) {
    std::vector<Tensor> parts{GU * time_step};
    std::vector<double> diag = parts[0].abs().sum(-1, true).data();
    for (int eta = 1;;) {
        aux_get_apower(E, A, eta);
        const Tensor add =
            aux_power_tensor(E, A, eta).matmul(GU) * (std::pow(time_step, eta + 1) / (eta + 1));
        const std::vector<double> add_diag = add.abs().sum(-1, true).data();
        bool stop = true;
        for (std::size_t i = 0; i < diag.size(); ++i) {
            stop = stop && add_diag[i] <= machine_eps() * diag[i];
            diag[i] += add_diag[i];
        }
        parts.push_back(add);
        if (stop) break;
        if (++eta == kMaxEta) return std::nullopt;
    }
    return Tensor::concat(parts, -1);
}

// Specifications ----------------------------------------------------------------------------------

/// The halfspaces of the specifications as safe sets C y <= d with unit normals: an unsafe
/// set a'y <= b is the safe set -a'y <= -b.
SpecRows aux_spec_rows(const std::vector<Specification> &specs) {
    SpecRows rows;
    // A halfspace with a unit normal: scaling does not change which points satisfy it.
    auto add = [&](const Halfspace &h, double sign) {
        const std::vector<double> a = h.a.data(); // a nonzero column of p rows (checked by verify)
        double norm = 0;
        for (double v : a) norm += v * v;
        norm = std::sqrt(norm);
        for (double v : a) rows.Cs.push_back(sign * v / norm);
        rows.ds.push_back(sign * h.b / norm);
        ++rows.nr;
    };
    for (const Specification &spec : specs) {
        // Each halfspace of a safe set is a row of its own: the set holds if all of them do.
        switch (spec.type()) {
        case SpecType::safe_set:
            for (const Halfspace &h : spec.halfspaces()) add(h, 1.0);
            break;
        case SpecType::unsafe_set:
            add(spec.halfspaces()[0], -1.0); // one halfspace (checked by verify)
            break;
        default:
            CORA_FAIL(Error::wrong_value, "LinearSys::verify: unknown specification type");
        }
    }
    return rows;
}

/// The largest step size <= time_step0 that hits t_final with a whole number of steps.
double aux_time_step(double time_step0, double t_final) {
    return t_final / std::ceil(t_final / time_step0);
}

/// Removes the times [t0, t1] from the unverified intervals, dropping slivers (VerifyTime).
void aux_remove_from_unsat(Intervals &unsat, double t0, double t1) {
    VerifyTime times(std::move(unsat));
    times.setdiff(t0, t1);
    unsat = std::move(times.bounds);
}

/// The initial state c + G sign(l'G) of R0 that reaches the support value of row `row` of ltg.
Tensor aux_extreme_x0(const Zonotope &R0, const std::vector<double> &ltg, std::size_t row) {
    const int64_t m0 = R0.generators().shape()[1];
    std::vector<double> sgn(m0);
    for (int64_t j = 0; j < m0; ++j) {
        const double v = ltg[row * m0 + j];
        sgn[j] = (v > 0) - (v < 0);
    }
    return R0.center() + R0.generators().matmul(Tensor::like(R0.center(), sgn, {m0, 1}));
}

// Propagation -------------------------------------------------------------------------------------

/// Propagates the directions for N steps and gathers the distances of the affine solution
/// (time points) and of the curvature enlargement (time intervals) to every halfspace.
Pass aux_pass(const Zonotope &R0, const Tensor &l_t, const std::vector<double> &ds, bool isu,
              const Tensor &Delta, const Tensor &Pu, const Tensor &u, const IntMat &F,
              const IntMat &G, int N) {
    const int S = static_cast<int>(ds.size());
    const int64_t m0 = R0.generators().shape()[1];
    Pass pass;
    std::vector<Tensor> rows{l_t};
    for (int k = 0; k < N; ++k) rows.push_back(rows.back().matmul(Delta));
    pass.LT = Tensor::concat(rows, -2);
    const std::vector<double> ltc = pass.LT.matmul(R0.center()).data();
    pass.ltg = pass.LT.matmul(R0.generators()).data();

    // The accumulated particular solution of the constant input, projected on l.
    std::vector<double> PuProj;
    if (isu && N > 0) {
        std::vector<Tensor> cols;
        Tensor P = Pu.zeros_like();
        cols.reserve(N);
        for (int k = 0; k < N; ++k) cols.push_back(P = Delta.matmul(P) + Pu);
        PuProj = l_t.matmul(Tensor::concat(cols, -1)).data();
    }
    pass.affine_tp.assign(std::size_t(N + 1) * S, 0.0);
    for (int k = 0; k <= N; ++k)
        for (int s = 0; s < S; ++s) {
            const std::size_t r = std::size_t(k) * S + s;
            double sum_abs = 0;
            for (int64_t j = 0; j < m0; ++j) sum_abs += std::abs(pass.ltg[r * m0 + j]);
            const double Pus = (isu && k > 0) ? PuProj[std::size_t(s) * N + (k - 1)] : 0.0;
            pass.affine_tp[r] = -ds[s] + ltc[r] + sum_abs + Pus;
        }

    // The curvature term F R0 + G u, evaluated along l_k for k < N.
    const Tensor cx = F.center.matmul(R0.center()), G1x = F.center.matmul(R0.generators());
    const Tensor G2x =
        F.rad.matmul(Tensor::concat({R0.center(), R0.generators()}, -1).abs().sum(-1, true));
    std::vector<Tensor> W{cx, G2x};
    if (isu) {
        W.push_back(G.center.matmul(u));
        W.push_back(G.rad.matmul(u.abs()));
    }
    const int64_t nw = W.size();
    const std::vector<double> lw = pass.LT.matmul(Tensor::concat(W, -1)).data();
    const std::vector<double> lg1 = pass.LT.matmul(G1x).data();
    pass.Cbloat.assign(std::size_t(N) * S, 0.0);
    for (std::size_t r = 0; r < pass.Cbloat.size(); ++r) {
        double sum_abs = 0;
        for (int64_t j = 0; j < m0; ++j) sum_abs += std::abs(lg1[r * m0 + j]);
        double c = lw[r * nw] + sum_abs + std::abs(lw[r * nw + 1]);
        if (isu) c = (c + lw[r * nw + 2]) + std::abs(lw[r * nw + 3]);
        pass.Cbloat[r] = c;
    }
    return pass;
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult priv_verify_ra_support_function(const LinearSys &sys, const Zonotope &R0,
                                             const Zonotope &U, double t_final,
                                             const std::vector<Specification> &specs) {
    const auto start = std::chrono::steady_clock::now();
    const FlushDenormals flush;
    VerifyResult res;
    auto finish = [&]() {
        res.computation_time = priv_seconds_since(start);
        return res;
    };

    // The system and the input: x' = A x + u with u in B U, split into the constant
    // u_trans = B center(U) and the uncertain part GU = B generators(U).
    const Tensor &A = sys.A();
    const CanonicalInput input = priv_canonical_input(sys, U);
    const int64_t p = input.p;
    const Tensor &u = input.u_trans, &GU = input.GU;
    const bool is_u = GU.shape()[1] > 0;
    const bool isu = u.abs().sum().item() != 0;

    // The directions l = (Cs C)' of the support function evaluations, per halfspace.
    const SpecRows rows = aux_spec_rows(specs);
    const int S = rows.nr;
    Tensor CsT = Tensor::like(A, rows.Cs, {S, p});
    const Tensor l_t = sys.C() ? CsT.matmul(*sys.C()) : CsT; // (S, n)
    const std::vector<double> &ds = rows.ds;

    Expmat E;
    E.Apower.push_back(A.data());
    aux_init_sparse(E, A);
    if (isu || is_u) E.Ainv = priv_inverse(A);

    // Initial set ---------------------------------------------------------------------------------
    // Distance of the initial output set to each halfspace; > 0 means it is already violated.
    {
        const std::vector<double> lc = l_t.matmul(R0.center()).data();
        const std::vector<double> lg = l_t.matmul(R0.generators()).data();
        const int64_t m0 = R0.generators().shape()[1];
        for (int s = 0; s < S; ++s) {
            double sum_abs = 0;
            for (int64_t j = 0; j < m0; ++j) sum_abs += std::abs(lg[s * m0 + j]);
            if (-ds[s] + lc[s] + sum_abs > 0) {
                res.falsification = Falsification{aux_extreme_x0(R0, lg, s), 0.0};
                return finish();
            }
        }
    }

    // Adaptive loop -------------------------------------------------------------------------------
    double time_step = aux_time_step(t_final / kStepsStart, t_final);
    std::vector<Intervals> unsat(S, Intervals{{0.0, t_final}});
    while (true) {
        ++res.iterations;
        const int N = static_cast<int>(std::round(t_final / time_step));
        res.time_step = time_step;
        res.num_steps = N;
        const Tensor Delta = aux_expm(A, E, time_step);

        // The constant input solution and the interval matrices; a slow sum shrinks the step.
        Tensor Pu = u.zeros_like();
        if (isu) {
            const std::optional<Tensor> P = aux_particular(A, u, E, Delta, time_step);
            if (!P) {
                time_step *= kShrinkNonconverged;
                continue;
            }
            Pu = *P;
        }
        IntMat F, G;
        if (!aux_intmat(A, isu, E, time_step, F, G)) {
            time_step *= kShrinkNonconverged;
            continue;
        }

        // The affine solution at the time points: a positive distance is a hit by the
        // trajectory from the extreme point of R0 with the centre of U.
        const Pass pass = aux_pass(R0, l_t, ds, isu, Delta, Pu, u, F, G, N);
        for (int k = 1; k <= N; ++k)
            for (int s = 0; s < S; ++s)
                if (pass.affine_tp[std::size_t(k) * S + s] > 0) {
                    res.falsification = Falsification{
                        aux_extreme_x0(R0, pass.ltg, std::size_t(k) * S + s), time_step * k};
                    return finish();
                }

        // Outer distance over each step: the larger end point plus curvature and, with an
        // uncertain input, the outer approximation of its particular solution.
        std::vector<double> over_pu(std::size_t(N + 1) * S, 0.0);
        if (is_u) {
            const std::optional<Tensor> under_g = aux_particular(A, GU, E, Delta, time_step);
            if (!under_g) {
                time_step *= kShrinkNonconverged;
                continue;
            }
            const std::optional<Tensor> over_g = aux_over_pu(A, GU, E, time_step);
            if (!over_g) {
                time_step *= kShrinkNonconverged;
                continue;
            }
            // Both sums accumulate over the steps; index k is the solution at t_k.
            std::vector<double> under_pu(std::size_t(N + 1) * S, 0.0);
            const std::vector<double> d_under = pass.LT.matmul(*under_g).abs().sum(-1, true).data();
            const std::vector<double> d_over = pass.LT.matmul(*over_g).abs().sum(-1, true).data();
            for (int k = 1; k <= N; ++k)
                for (int s = 0; s < S; ++s) {
                    const std::size_t r = std::size_t(k) * S + s, prev = r - S;
                    under_pu[r] = under_pu[prev] + d_under[prev];
                    over_pu[r] = over_pu[prev] + d_over[prev];
                }
            // The inner approximation of the solution at the time points reaches a halfspace.
            for (int k = 0; k <= N; ++k)
                for (int s = 0; s < S; ++s) {
                    const std::size_t r = std::size_t(k) * S + s;
                    if (pass.affine_tp[r] + under_pu[r] > 0) {
                        res.falsification =
                            Falsification{aux_extreme_x0(R0, pass.ltg, r), time_step * k};
                        return finish();
                    }
                }
        }

        // Time intervals whose outer distance is below 0 for every halfspace are verified.
        std::vector<std::vector<double>> worst(S, std::vector<double>(N));
        bool all_neg = true;
        for (int s = 0; s < S; ++s) {
            for (int k = 0; k < N; ++k) {
                const std::size_t r = std::size_t(k) * S + s, next = r + S;
                const double a = pass.affine_tp[r] + pass.Cbloat[r];
                const double b = pass.affine_tp[next] + pass.Cbloat[r];
                worst[s][k] = std::max(a, b) + over_pu[next];
                all_neg = all_neg && worst[s][k] < 0;
            }
        }
        for (int s = 0; s < S; ++s) {
            bool spec_neg = true;
            for (double w : worst[s]) spec_neg = spec_neg && w < 0;
            // Without an uncertain input the verified set is cleared only if all are verified.
            if (is_u ? spec_neg : all_neg) {
                unsat[s].clear();
                continue;
            }
            double t0 = 0, t1 = time_step;
            for (int k = 0; k < N; ++k) {
                if (worst[s][k] < 0) aux_remove_from_unsat(unsat[s], t0, t1);
                t0 += time_step;
                t1 += time_step;
            }
        }

        // Without an uncertain input, the horizon shrinks to the last unverified time.
        bool all_verified = true;
        double last_unsat = 0;
        for (const Intervals &iv : unsat)
            if (!iv.empty()) {
                all_verified = false;
                last_unsat = std::max(last_unsat, iv.back()[1]);
            }
        if (all_verified) break;
        if (!is_u) t_final = last_unsat;
        time_step *= kShrinkFixed;
        if (time_step < numerics::kMinTimeStep) return finish();
    }
    res.verified = true;
    return finish();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
