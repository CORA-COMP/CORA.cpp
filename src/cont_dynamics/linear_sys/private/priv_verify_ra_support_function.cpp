// priv_verify_ra_support_function - reach-avoid verification of a linear system by supports
//
// Quicker than reachability: the set is never computed, only its distance to each halfspace
// C_s y <= d_s (every specification is turned into a safe set), by the support function of the
// affine solution along the back-propagated directions l_k = (e^{A dt}')^k (C_s C)'. A step size
// is tried, the distances are checked, and the step shrinks until every halfspace holds for the
// whole horizon or a trajectory of R0 and the centre of U hits one. Supported as in the ARCH
// benchmarks: x' = A x + B u, y = C x, a zonotope R0 and a constant zonotope U, no offsets.
//
// A homogeneous system whose growth is bounded (an accepted energy weight of a second-order system,
// priv_verify_energy.h, or mu t_final <= 1, mu its balanced log norm) runs CORALean's sound
// LinearSys.verifySupportFuncFloat, bit for bit on the lean backend: e^{A dt} with a radius, the
// rows' errors carried, exact times. The others keep CORA's propagation with CORALean's sound F, G
// and the support |l|'v of the boxes F_r |X0|, G_r |u|. Statistics: the iterations, the last step
// size and count, computation_time, the falsifying initial state and time.
//
// Syntax:   res = priv_verify_ra_support_function(sys, params, specs);
// Inputs:   sys - linear system;  params - R0, U, t_final
//           specs - safe and unsafe halfspaces
// Outputs:  res - verified (false if undecided at a step below 1e-12), and the run's statistics
// See also: LinearSys::verify
//
// CORA: contDynamics/@linearSys/private/priv_verifyRA_supportFunc.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_float_scalars.h"
#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cont_dynamics/linear_sys/private/priv_verify_correction.h"
#include "cont_dynamics/linear_sys/private/priv_verify_energy.h"
#include "cont_dynamics/linear_sys/private/priv_verify_exp_ball.h"
#include "cont_dynamics/linear_sys/private/priv_verify_time.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/threads.h"
#include "global/exact_time.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
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

/// A^i / i!, kept over the passes (CORALean's verifyPowerFloat); A^-1 if A is invertible.
struct Expmat {
    std::optional<VerifyPowers> powers;
    std::vector<double> scales; // balanceScalesFloat of A, for the remainders of F and G
    std::optional<Tensor> Ainv;
    // A by rows (CSR) when it is sparse: the powers then cost n nnz(A) instead of n^3.
    bool sparse_a = false;
    Csr a;
};

/// A matrix by rows, each stored from its first to its last nonzero column.
struct Banded {
    int64_t n = 0;
    std::vector<int64_t> lo, hi, offset; // the columns [lo, hi) of row k start at val[offset[k]]
    std::vector<double> val;
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

/// x B / d for a dense x (n, n); `empty(k)` says row k of B has no entries and `add(k, xik, row)`
/// adds x_ik times it to a row of the result. Each entry adds its products in the order of k, but
/// a block of 8 rows reads a row of B once, from the cache, and the blocks run in parallel.
template <class Empty, class Add>
std::vector<double> aux_mul_rows(const std::vector<double> &x, int64_t n, double d,
                                 const Empty &empty, const Add &add) {
    constexpr int64_t kRows = 8;
    std::vector<double> out(n * n, 0.0);
    const int64_t blocks = (n + kRows - 1) / kRows;
    [[maybe_unused]] const int threads = threads_for(blocks, n * n);
#pragma omp parallel for num_threads(threads) schedule(dynamic) if (threads > 1)
    for (int64_t blk = 0; blk < blocks; ++blk) {
        [[maybe_unused]] const FlushDenormals flush; // the flag is per thread
        const int64_t i0 = blk * kRows, i1 = std::min(n, i0 + kRows);
        for (int64_t k = 0; k < n; ++k) {
            if (empty(k)) continue;
            for (int64_t i = i0; i < i1; ++i) {
                const double xik = x[i * n + k];
                if (xik != 0) add(k, xik, &out[i * n]);
            }
        }
        for (int64_t e = i0 * n; e < i1 * n; ++e) out[e] /= d;
    }
    return out;
}

/// x B / d for a dense x (n, n) and a sparse B; B's rows skip the zero entries of x.
std::vector<double> aux_mul_csr(const std::vector<double> &x, const Csr &B, int64_t n, double d) {
    return aux_mul_rows(
        x, n, d, [&](int64_t k) { return B.row_ptr[k] == B.row_ptr[k + 1]; },
        [&](int64_t k, double xik, double *row) {
            for (int64_t q = B.row_ptr[k]; q < B.row_ptr[k + 1]; ++q)
                row[B.col_idx[q]] += xik * B.val[q];
        });
}

/// The rows of the (n, n) matrix m by band; the stored share of the entries is the density.
Banded aux_banded(const std::vector<double> &m, int64_t n) {
    Banded b;
    b.n = n;
    b.lo.assign(n, 0);
    b.hi.assign(n, 0);
    b.offset.assign(n + 1, 0);
    for (int64_t k = 0; k < n; ++k) {
        const double *row = &m[k * n];
        int64_t lo = 0, hi = n;
        while (lo < hi && row[lo] == 0) ++lo;
        while (hi > lo && row[hi - 1] == 0) --hi;
        if (lo == hi) lo = hi = 0;
        b.lo[k] = lo;
        b.hi[k] = hi;
        b.val.insert(b.val.end(), row + lo, row + hi);
        b.offset[k + 1] = static_cast<int64_t>(b.val.size());
    }
    return b;
}

/// x B / d for a dense x (n, n) and B by band: contiguous rows, which the compiler vectorizes;
/// the zeros inside a band add zeros to the entries.
std::vector<double> aux_mul_band(const std::vector<double> &x, const Banded &B, int64_t n,
                                 double d) {
    return aux_mul_rows(
        x, n, d, [&](int64_t k) { return B.lo[k] == B.hi[k]; },
        [&](int64_t k, double xik, double *row) {
            const double *v = &B.val[B.offset[k]] - B.lo[k];
            for (int64_t j = B.lo[k]; j < B.hi[k]; ++j) row[j] += xik * v[j];
        });
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
        const Csr c = aux_csr(sum, n);
        const Banded b = aux_banded(sum, n);
        sum = b.val.size() < 3 * c.val.size() ? aux_mul_band(sum, b, n, 1.0)
                                              : aux_mul_csr(sum, c, n, 1.0);
        ++squared;
    }
    Tensor result = Tensor::like(A, sum, {n, n});
    for (; squared < s; ++squared) result = result.matmul(result);
    return result;
}

/// A^i / i! as a tensor.
Tensor aux_power_tensor(Expmat &E, const Tensor &A, int i) {
    const int64_t n = A.shape()[0];
    return Tensor::like(A, E.powers->power(i), {n, n});
}

/// An interval matrix of CORALean's bounds as CORA's center and radius.
IntMat aux_center_radius(const Tensor &A, const VerifyCorrection &M) {
    const int64_t n = A.shape()[0];
    std::vector<double> rad(M.inf.size()), cen(M.inf.size());
    for (std::size_t i = 0; i < rad.size(); ++i) {
        rad[i] = 0.5 * (M.sup[i] - M.inf[i]);
        cen[i] = M.inf[i] + rad[i];
    }
    return {Tensor::like(A, cen, {n, n}), Tensor::like(A, rad, {n, n})};
}

/// The interval matrices F (state) and G (input, if isu) of the curvature enlargement: CORALean's
/// verifyCorrectionStateFloat and verifyCorrectionInputFloat, CORA's Taylor sums with their stop
/// and a remainder; false if a sum needs more than kMaxEta - 1 terms or its remainder fails.
bool aux_intmat(const Tensor &A, bool isu, Expmat &E, double time_step, IntMat &F, IntMat &G) {
    const std::optional<VerifyCorrection> f =
        priv_verify_correction_state(*E.powers, E.scales, time_step);
    if (!f) return false;
    F = aux_center_radius(A, *f);
    if (!isu) {
        G = IntMat{A.zeros_like(), A.zeros_like()};
        return true;
    }
    const std::optional<VerifyCorrection> g =
        priv_verify_correction_input(*E.powers, E.scales, time_step);
    if (!g) return false;
    G = aux_center_radius(A, *g);
    return true;
}

/// The matrix X of a particular solution A^-1 (e^{A dt} - I) X, by the series
/// sum_j A^{j-1} dt^j / j! until it stops changing if A is singular; empty if it diverges.
std::optional<Tensor> aux_particular(const Tensor &A, const Tensor &X, Expmat &E,
                                     const Tensor &Delta, double time_step) {
    if (E.Ainv) return E.Ainv->matmul((Delta - A.eye_like()).matmul(X));
    Tensor Asum = A.eye_like() * time_step;
    for (int eta = 2;; ++eta) {
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
    std::vector<double> diag = parts[0].abs_sum().data();
    for (int eta = 1;;) {
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

/// The halfspaces of the specifications as safe sets C y <= d, with unit normals if normalize (the
/// sound algorithm keeps the rows, so the halfspace is exactly the specification's): an unsafe
/// set a'y <= b is the safe set -a'y <= -b.
SpecRows aux_spec_rows(const std::vector<Specification> &specs, bool normalize = true) {
    SpecRows rows;
    // a halfspace with a unit normal: scaling does not change which points satisfy it
    auto add = [&](const SpecRow &h, double sign) {
        const std::vector<double> &a = h.a; // a nonzero row of p entries (checked by verify)
        double norm = 0;
        for (double v : a) norm += v * v;
        norm = normalize ? std::sqrt(norm) : 1.0;
        for (double v : a) rows.Cs.push_back(sign * v / norm);
        rows.ds.push_back(sign * h.b / norm);
        ++rows.nr;
    };
    for (const Specification &spec : specs) {
        const std::vector<SpecRow> halfspaces = priv_spec_rows(spec);
        // Each halfspace of a safe set is a row of its own: the set holds if all of them do.
        switch (spec.type()) {
        case SpecType::safe_set:
            for (const SpecRow &h : halfspaces) add(h, 1.0);
            break;
        case SpecType::unsafe_set:
            add(halfspaces[0], -1.0); // one halfspace (checked by verify)
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

// Propagation of the directions -------------------------------------------------------------------

/// The rows l_0 = l0, l_{k+1} = l_k D for k < N of one direction, row-major (N + 1, n), each
/// entry summed as the backend's dense product (1, n) x (n, n) sums it: blocks of rows of D,
/// the terms of a block in order from zero, the blocks added in order; zero terms are skipped.
std::vector<double> aux_propagate(const std::vector<double> &l0, const Banded &D, int N,
                                  bool flush = true) {
    const int64_t n = D.n;
    const int64_t block = n < 128 ? n : (n * 8 < 32000 ? 16 : 4);
    std::vector<double> lt(std::size_t(N + 1) * n, 0.0);
    std::copy(l0.begin(), l0.end(), lt.begin());
    [[maybe_unused]] const int threads = threads_for(n, static_cast<long long>(D.val.size()));
#pragma omp parallel num_threads(threads) if (threads > 1)
    {
        std::optional<FlushDenormals> flushing; // the flag is per thread
        if (flush) flushing.emplace();
#ifdef _OPENMP
        const int64_t t = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        const int64_t t = 0, nt = 1;
#endif
        const int64_t c0 = n * t / nt, c1 = n * (t + 1) / nt;
        std::vector<double> sum(c1 - c0);
        // one step: every thread sums its own columns of the next row from the row before
        for (int step = 0; step < N; ++step) {
            const double *x = &lt[std::size_t(step) * n];
            double *y = &lt[std::size_t(step + 1) * n];
            // the block size of Eigen's gemv: 16 rows of D (4 from n = 4000, all below n = 128)
            for (int64_t k0 = 0; k0 < n; k0 += block) {
                std::fill(sum.begin(), sum.end(), 0.0);
                for (int64_t k = k0; k < std::min(n, k0 + block); ++k) {
                    const double xk = x[k];
                    const int64_t j0 = std::max(c0, D.lo[k]), j1 = std::min(c1, D.hi[k]);
                    if (xk == 0 || j0 >= j1) continue;
                    const double *v = &D.val[D.offset[k] - D.lo[k]];
                    for (int64_t j = j0; j < j1; ++j) sum[j - c0] += xk * v[j];
                }
                for (int64_t j = c0; j < c1; ++j) y[j] += sum[j - c0];
            }
            // the next row needs this one whole
#pragma omp barrier
        }
    }
    return lt;
}

// Propagation -------------------------------------------------------------------------------------

/// Propagates the directions for N steps and gathers the distances of the affine solution
/// (time points) and of the curvature enlargement (time intervals) to every halfspace.
Pass aux_pass(const Zonotope &R0, const Tensor &l_t, const std::vector<double> &ds, bool isu,
              const Tensor &Delta, const Tensor &Pu, const Tensor &u, const IntMat &F,
              const IntMat &G, int N, bool host_rows) {
    const int S = static_cast<int>(ds.size());
    const int64_t m0 = R0.generators().shape()[1];
    Pass pass;
    const int64_t n = Delta.shape()[0];
    // one direction of a sparse system on Eigen: the banded rows of Delta on the host, in threads
    const bool host = host_rows && Delta.products_as_eigen();
    const Banded D = (S == 1 && n >= 128 && host) ? aux_banded(Delta.data(), n) : Banded{};
    const bool wide = !D.val.empty() && threads_for(n, static_cast<long long>(D.val.size())) > 1;
    if (!D.val.empty() && (wide || double(D.val.size()) < 0.6 * double(n) * double(n))) {
        pass.LT = Tensor::like(l_t, aux_propagate(l_t.data(), D, N), {N + 1, n});
    } else {
        std::vector<Tensor> rows{l_t};
        for (int k = 0; k < N; ++k) rows.push_back(rows.back().matmul(Delta));
        pass.LT = Tensor::concat(rows, -2);
    }
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
    // the boxes F_r |X0| and G_r |u| about the origin have the support |l|' v, not |l' v| (CORA)
    std::vector<Tensor> Wc{cx}, Wr{G2x};
    if (isu) {
        Wc.push_back(W[2]);
        Wr.push_back(W[3]);
    }
    const int64_t nw = static_cast<int64_t>(Wc.size());
    const std::vector<double> lw = pass.LT.matmul(Tensor::concat(Wc, -1)).data();
    const std::vector<double> lr = pass.LT.abs().matmul(Tensor::concat(Wr, -1)).data();
    const std::vector<double> lg1 = pass.LT.matmul(G1x).data();
    pass.Cbloat.assign(std::size_t(N) * S, 0.0);
    for (std::size_t r = 0; r < pass.Cbloat.size(); ++r) {
        double sum_abs = 0;
        for (int64_t j = 0; j < m0; ++j) sum_abs += std::abs(lg1[r * m0 + j]);
        double c = lw[r * nw] + sum_abs + lr[r * nw];
        if (isu) c = (c + lw[r * nw + 1]) + lr[r * nw + 1];
        pass.Cbloat[r] = c;
    }
    // per row the bound is the center term, the absolute generator sums and the particular terms
    return pass;
}

// The sound pass ----------------------------------------------------------------------------------
// Sound in the nearest model, as the oracle's `nearest`: no rounding terms, so FTZ is no loss.

constexpr int kStepsDiv = 100; // the first step size is t_final / ceil(t_final / (t_final/100))
constexpr double kBoundLimit = 536870912.0; // 2^29: the largest exponent of the growth bound

using ExactIntervals = std::vector<std::array<ExactTime, 2>>;

/// sum_{j = n-1 .. 0} f(j), each term added in front of the sum of the later ones.
template <class F>
double aux_sum_up(int64_t n, const F &f) {
    double sum = 0;
    for (int64_t j = n - 1; j >= 0; --j) sum = f(j) + sum;
    return sum;
}

// The input's boxes -------------------------------------------------------------------------------

/// The input in state space, B u_c and B G_U (row-major (n, k)), as
/// LinearSys.verifySupportFuncFloat brackets them (points at nearest), and whether either is not
/// zero.
struct SoundInput {
    std::vector<double> u0, M;
    int64_t k = 0;
    bool has_u = false, has_m = false;
};

/// What a run keeps over its passes: the scales, the growth bound kappa e^{t mu}, the initial set's
/// bounds, the rows C_s C and their errors (0 at nearest), the input.
struct SoundRun {
    std::vector<double> s, v, d, E0;
    double kappa = 1, mu = 0, beta_x = 0;
    Tensor L; // (S, n)
    int S = 0;
    SoundInput in;
};

/// One step's input at dt (LinearSys.verifyInputSolutionFloat and the box of G u0): the box of w =
/// (int_0^dt e^{sA} ds) u0, the orders P_i M (i < eta, each (n, k)) and their sum, the remainder
/// rho, the box of G u0, and the bounds beta of their scaled row sums.
struct SoundInputBoxes {
    std::vector<double> w_lo, w_hi, Vs, rho, g_lo, g_hi;
    std::vector<std::vector<double>> V;
    double beta_w = 0, beta_g = 0, beta_u = 0;
};

// The support bounds ------------------------------------------------------------------------------

/// The support bounds of X0 = <c, G> along the rows `ltc`, `ltg` (rows of LT times c and G), with
/// the errors E: (lc + sum_j |lg_j|) + E beta and (lc + sum_j |lg_j|) - E beta.
struct SupportBounds {
    std::vector<double> up, down;
};
SupportBounds aux_support_bounds(const std::vector<double> &ltc, const std::vector<double> &ltg,
                                 int64_t m0, const std::vector<double> &E, double beta) {
    SupportBounds b{std::vector<double>(ltc.size()), std::vector<double>(ltc.size())};
    for (std::size_t r = 0; r < ltc.size(); ++r) {
        const double g = aux_sum_up(m0, [&](int64_t j) { return std::abs(ltg[r * m0 + j]); });
        const double base = ltc[r] + g;
        b.up[r] = base + E[r] * beta;
        b.down[r] = base - E[r] * beta;
    }
    return b;
}

/// CORALean's expUp 30 a at nearest: (1 + h + 2h^2)^(2^30) for h = a / 2^30 (halvings exact).
double aux_exp_up(double a) {
    constexpr int kDepth = 30;
    double h = a;
    for (int i = 0; i < kDepth; ++i) h *= 0.5;
    double u = (1.0 + h) + (h * h + h * h);
    for (int i = 0; i < kDepth; ++i) u = u * u;
    return u;
}

/// The removal of the times [t0, t1] from the unverified intervals (LinearSys.verifyTimeSetdiff):
/// exact, and nothing dropped.
void aux_setdiff_exact(ExactIntervals &U, const ExactTime &t0, const ExactTime &t1) {
    ExactIntervals kept;
    for (const auto &iv : U) {
        if (t1 <= iv[0] || iv[1] <= t0) {
            kept.push_back(iv);
            continue;
        }
        if (iv[0] < t0) kept.push_back({iv[0], t0});
        if (t1 < iv[1]) kept.push_back({t1, iv[1]});
    }
    U = std::move(kept);
}

/// The smallest N with N dt >= tf, exactly.
int64_t aux_steps_to(const ExactTime &tf, double dt) {
    auto N = static_cast<int64_t>(std::ceil(tf.nearest() / dt));
    const ExactTime step(dt);
    while (step.times(static_cast<std::uint32_t>(N)) < tf) ++N;
    while (N > 0 && tf <= step.times(static_cast<std::uint32_t>(N - 1))) --N;
    return N;
}

// The input's series ------------------------------------------------------------------------------

/// max(0, max_l x_l / s_l): CORALean's maximum zero (maximumOver fun l => divUp (x l) (s l)).
double aux_scaled_max(const std::vector<double> &x, const std::vector<double> &s) {
    double beta = 0;
    for (std::size_t l = 0; l < x.size(); ++l) beta = std::max(beta, x[l] / s[l]);
    return beta;
}

/// LinearSys.step_verifySupportInputBoxesFloat at nearest: empty if a series does not stop by
/// order 74 or a remainder is not admissible. Order i is [dt^{i+1}/(i+1)] A^i/i! applied to u0 and
/// M; zero terms of a sum are skipped (they change no sum that starts at +0).
std::optional<SoundInputBoxes> aux_input_boxes(Expmat &E, const SoundRun &run, int64_t n,
                                               double dt) {
    const SoundInput &in = run.in;
    const int64_t k = in.k;
    const std::size_t nk = std::size_t(n) * k;
    SoundInputBoxes b;
    b.w_lo.assign(n, 0.0);
    b.w_hi.assign(n, 0.0);
    b.Vs.assign(nk, 0.0);
    b.rho.assign(n, 0.0);
    b.g_lo.assign(n, 0.0);
    b.g_hi.assign(n, 0.0);
    if (in.has_u || in.has_m) {
        constexpr int kOrders = 75; // the search's fuel: orders 0 to 74
        const double eps = machine_eps();
        // the nonzeros of u0 and of each column of M, from the last index down
        std::vector<int64_t> nzu;
        for (int64_t j = n - 1; j >= 0; --j)
            if (in.u0[j] != 0) nzu.push_back(j);
        std::vector<std::vector<int64_t>> nzm(k);
        for (int64_t j = n - 1; j >= 0; --j)
            for (int64_t c = 0; c < k; ++c)
                if (in.M[std::size_t(j) * k + c] != 0) nzm[c].push_back(j);
        std::vector<double> S(n, 0.0), wsum(n, 0.0);
        int eta = -1;
        for (int i = 0; i < kOrders && eta < 0; ++i) {
            // the coefficient dt^{i+1}/(i+1) and the power A^i/i! (the identity at i = 0)
            const double coef = priv_float_power(i + 1, dt) / double(i + 1);
            const std::vector<double> *P = i == 0 ? nullptr : &E.powers->power(i);
            auto entry = [&](int64_t l, int64_t j) {
                return P ? coef * (*P)[std::size_t(l) * n + j] : (l == j ? coef * 1.0 : 0.0);
            };
            // [P_i] u0 and [P_i] M, each entry summed from the last index down
            std::vector<double> tw(n, 0.0), tv(nk, 0.0), a(n);
            for (int64_t l = 0; l < n; ++l) {
                double sum = 0;
                for (const int64_t j : nzu) sum = entry(l, j) * in.u0[j] + sum;
                tw[l] = sum;
                double rs = 0; // sumUp over the columns of |[P_i M]_lc|, from the last
                for (int64_t c = k - 1; c >= 0; --c) {
                    double v = 0;
                    for (const int64_t j : nzm[c])
                        v = entry(l, j) * in.M[std::size_t(j) * k + c] + v;
                    tv[std::size_t(l) * k + c] = v;
                    rs = std::abs(v) + rs;
                }
                a[l] = std::abs(tw[l]) + rs;
            }
            // CORA's stop: the order's row sums at most eps times those of the orders before it
            bool stop = i >= 1;
            for (int64_t l = 0; l < n && stop; ++l) stop = a[l] <= eps * S[l];
            if (stop) {
                eta = i;
                break;
            }
            for (int64_t l = 0; l < n; ++l) {
                S[l] = S[l] + a[l];
                wsum[l] = wsum[l] + tw[l];
            }
            for (std::size_t e = 0; e < nk; ++e) b.Vs[e] = b.Vs[e] + tv[e];
            b.V.push_back(std::move(tv));
        }
        if (eta < 0) return std::nullopt;
        // the remainder past the orders i < eta: dt W |u0| and dt W sum_c |M_c|
        const std::optional<std::vector<double>> W =
            priv_verify_scaled_remainder(E.powers->A(), E.scales, n, dt, eta - 1);
        if (!W) return std::nullopt;
        std::vector<double> m_abs(n);
        for (int64_t j = 0; j < n; ++j) {
            double rs = 0;
            for (int64_t c = k - 1; c >= 0; --c) rs = std::abs(in.M[std::size_t(j) * k + c]) + rs;
            m_abs[j] = rs;
        }
        for (int64_t l = 0; l < n; ++l) {
            const double *Wl = &(*W)[std::size_t(l) * n];
            const double ru =
                dt * aux_sum_up(n, [&](int64_t j) { return Wl[j] * std::abs(in.u0[j]); });
            b.rho[l] = dt * aux_sum_up(n, [&](int64_t j) { return Wl[j] * m_abs[j]; });
            b.w_lo[l] = wsum[l] - ru;
            b.w_hi[l] = wsum[l] + ru;
        }
    }
    // G u0, the box of the constant input's curvature term
    if (in.has_u) {
        const std::optional<VerifyCorrection> G =
            priv_verify_correction_input(*E.powers, E.scales, dt);
        if (!G) return std::nullopt;
        for (int64_t l = 0; l < n; ++l) {
            double lo = 0, hi = 0;
            for (int64_t j = n - 1; j >= 0; --j) {
                const double u = in.u0[j];
                if (u == 0) continue;
                const double p = G->inf[std::size_t(l) * n + j] * u;
                const double q = G->sup[std::size_t(l) * n + j] * u;
                lo = std::min(p, q) + lo;
                hi = std::max(p, q) + hi;
            }
            b.g_lo[l] = lo;
            b.g_hi[l] = hi;
        }
    }
    // the bounds of the scaled row sums of w, G u0, and the orders with the remainder
    std::vector<double> mw(n), mg(n), mu(n);
    for (int64_t l = 0; l < n; ++l) {
        mw[l] = std::max(std::abs(b.w_lo[l]), std::abs(b.w_hi[l])) + 0.0;
        mg[l] = std::max(std::abs(b.g_lo[l]), std::abs(b.g_hi[l])) + 0.0;
        double acc = 0;
        for (const std::vector<double> &Vi : b.V) {
            double rs = 0;
            for (int64_t c = k - 1; c >= 0; --c) rs = std::abs(Vi[std::size_t(l) * k + c]) + rs;
            acc = acc + rs;
        }
        mu[l] = acc + b.rho[l];
    }
    b.beta_w = aux_scaled_max(mw, run.s);
    b.beta_g = aux_scaled_max(mg, run.s);
    b.beta_u = aux_scaled_max(mu, run.s);
    return b;
}

/// sum_{l = n-1 .. 0} max(x_l lo_l, x_l hi_l) and the same with min: the dot product of a row with
/// a box, from above and below.
std::array<double, 2> aux_box_dot(const double *x, const std::vector<double> &lo,
                                  const std::vector<double> &hi, int64_t n) {
    double up = 0, dn = 0;
    for (int64_t l = n - 1; l >= 0; --l) {
        const double a = x[l] * lo[l], b = x[l] * hi[l];
        up = std::max(a, b) + up;
        dn = std::min(a, b) + dn;
    }
    return {up, dn};
}

// One sound pass ----------------------------------------------------------------------------------

/// The outcome of a sound pass.
struct SoundPass {
    enum class Kind { not_converged, no_bound, falsified, decided } kind = Kind::decided;
    int64_t k = 0;                       // the falsifying time point
    int q = 0;                           // and halfspace
    std::vector<std::vector<char>> flag; // flag[q][k]: step k verified for halfspace q
    std::vector<double> ltg;             // the rows of LT G, for the falsifying initial state
};

/// LinearSys.verifySupportPassFloat: N steps of dt; falsification at k dt <= T only.
SoundPass aux_sound_pass(const Tensor &A, Expmat &E, const SoundRun &run, const Zonotope &R0,
                         double dt, int64_t N, const ExactTime &T, bool host_rows) {
    SoundPass out;
    const int64_t n = A.shape()[0];
    const int S = run.S;
    // the correction matrix F and the exponential with its radius, at dt
    const std::optional<VerifyCorrection> F = priv_verify_correction_state(*E.powers, E.scales, dt);
    if (!F) {
        out.kind = SoundPass::Kind::not_converged;
        return out;
    }
    const std::optional<VerifyExpBall> ball = priv_verify_exp_ball(A, E.scales, dt);
    if (!ball) {
        out.kind = SoundPass::Kind::not_converged;
        return out;
    }
    // the input's boxes at dt
    const std::optional<SoundInputBoxes> inb = aux_input_boxes(E, run, n, dt);
    if (!inb) {
        out.kind = SoundPass::Kind::not_converged;
        return out;
    }
    // F = [Fc +- Fr] and the products the curvature term needs
    const std::size_t nn = std::size_t(n) * n;
    std::vector<double> Fc(nn), Fr(nn), Fm(nn);
    for (std::size_t e = 0; e < nn; ++e) {
        Fc[e] = F->inf[e] + (F->sup[e] - F->inf[e]) * 0.5;
        Fr[e] = std::max(F->sup[e] - Fc[e], Fc[e] - F->inf[e]);
        Fm[e] = std::abs(Fc[e]) + Fr[e];
    }
    // the boxes F_c c, F_c G and F_r v, and beta_f >= |F| v / s
    const Tensor Fct = Tensor::like(A, Fc, {n, n});
    const Tensor wc = Fct.matmul(R0.center()), WG = Fct.matmul(R0.generators());
    const Tensor vt = Tensor::like(A, run.v, {n, 1});
    const Tensor wr = Tensor::like(A, Fr, {n, n}).matmul(vt);
    const std::vector<double> fv = Tensor::like(A, Fm, {n, n}).matmul(vt).data();
    double beta_f = 0;
    for (int64_t i = 0; i < n; ++i) beta_f = std::max(beta_f, fv[i] / run.s[i]);
    // the growth bound over N dt
    const double a = std::max(run.mu, 0.0) * (double(N) * dt);
    if (!(a <= kBoundLimit)) {
        out.kind = SoundPass::Kind::no_bound;
        return out;
    }
    // Y = max(1, kappa e^{mu+ N dt}), with CORALean's maximum
    const double ky = run.kappa * aux_exp_up(a);
    const double Y = 1.0 <= ky ? ky : (ky < 1.0 ? 1.0 : 1.0 + ky);
    // the rows l_k (k * S + q), propagated by D
    const Tensor &D = ball->D;
    const int64_t nd = D.shape()[0];
    const bool host = host_rows && D.products_as_eigen();
    const Banded Db = (S == 1 && nd >= 128 && host) ? aux_banded(D.data(), nd) : Banded{};
    Tensor LT;
    if (!Db.val.empty() && (threads_for(nd, static_cast<long long>(Db.val.size())) > 1 ||
                            double(Db.val.size()) < 0.6 * double(nd) * double(nd))) {
        LT = Tensor::like(run.L, aux_propagate(run.L.data(), Db, static_cast<int>(N), false),
                          {N + 1, n});
    } else {
        std::vector<Tensor> rows{run.L};
        for (int64_t k = 0; k < N; ++k) rows.push_back(rows.back().matmul(D));
        LT = Tensor::concat(rows, -2);
    }
    // along every row by l: c, G, F_c c, F_c G, P_i M and their sum; by |l|: F_r v, s and rho
    const int64_t m0 = R0.generators().shape()[1], mw = WG.shape()[1];
    const bool has_input = run.in.has_u || run.in.has_m;
    const int64_t k_in = has_input ? run.in.k : 0;
    const int64_t nv = has_input ? static_cast<int64_t>(inb->V.size()) : 0;
    std::vector<Tensor> cols{R0.center()};
    if (m0 > 0) cols.push_back(R0.generators());
    cols.push_back(wc);
    if (mw > 0) cols.push_back(WG);
    if (k_in > 0 && nv > 0) {
        // the orders side by side, order by order
        std::vector<double> cat(std::size_t(n) * k_in * nv);
        for (int64_t l = 0; l < n; ++l)
            for (int64_t i = 0; i < nv; ++i)
                for (int64_t c = 0; c < k_in; ++c)
                    cat[(std::size_t(l) * nv + i) * k_in + c] =
                        inb->V[std::size_t(i)][std::size_t(l) * k_in + c];
        cols.push_back(Tensor::like(A, cat, {n, nv * k_in}));
        cols.push_back(Tensor::like(A, inb->Vs, {n, k_in}));
    }
    const std::vector<double> lp = LT.matmul(Tensor::concat(cols, -1)).data();
    const Tensor LTabs = LT.abs();
    std::vector<Tensor> acols{wr, Tensor::like(A, run.s, {n, 1})};
    if (has_input) acols.push_back(Tensor::like(A, inb->rho, {n, 1}));
    const std::vector<double> la = LTabs.matmul(Tensor::concat(acols, -1)).data();
    // the columns of the two products, row by row
    const std::size_t rows = std::size_t(N + 1) * S;
    const int64_t wp = 1 + m0 + 1 + mw + (k_in > 0 && nv > 0 ? nv * k_in + k_in : 0);
    const int64_t wa = has_input ? 3 : 2;
    auto columns = [&](const std::vector<double> &P, int64_t width, int64_t from, int64_t count) {
        std::vector<double> out_cols(rows * std::size_t(count));
        for (std::size_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < count; ++j) out_cols[r * count + j] = P[r * width + from + j];
        return out_cols;
    };
    const std::vector<double> ltc = columns(lp, wp, 0, 1);
    out.ltg = columns(lp, wp, 1, m0);
    const std::vector<double> lw = columns(lp, wp, 1 + m0, 1);
    const std::vector<double> lg1 = columns(lp, wp, 2 + m0, mw);
    const std::vector<double> lv = columns(lp, wp, 2 + m0 + mw, k_in > 0 ? nv * k_in : 0);
    const std::vector<double> lvs = columns(lp, wp, 2 + m0 + mw + nv * k_in, k_in > 0 ? k_in : 0);
    const std::vector<double> lr = columns(la, wa, 0, 1);
    const std::vector<double> wn = columns(la, wa, 1, 1);
    const std::vector<double> lrho = has_input ? columns(la, wa, 2, 1) : std::vector<double>{};
    // the error sums acc_k = E0 + sum_{j<k} ||l_j||_s r and the errors Y acc_k
    std::vector<double> err(rows);
    {
        std::vector<double> acc = run.E0;
        for (int64_t k = 0; k <= N; ++k)
            for (int q = 0; q < S; ++q) {
                const std::size_t r = std::size_t(k) * S + q;
                err[r] = Y * acc[q];
                acc[q] = acc[q] + wn[r] * ball->r;
            }
    }
    const SupportBounds sb = aux_support_bounds(ltc, out.ltg, m0, err, run.beta_x);
    // the input along every row l_k, k < N: l'w, l'(G u0), the outer and inner support, summed
    const std::size_t steps = std::size_t(N) * S;
    std::vector<double> w_up(steps, 0.0), w_down(steps, 0.0), g_up(steps, 0.0), outer(steps, 0.0),
        inner(steps, 0.0);
    if (has_input) { // without an input every term is 0, as in CORALean
        const int64_t k = k_in;
        const std::vector<double> LTd = LT.data();
        [[maybe_unused]] const int threads = threads_for(static_cast<int64_t>(steps), n);
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
        for (int64_t ri = 0; ri < static_cast<int64_t>(steps); ++ri) {
            const auto r = static_cast<std::size_t>(ri);
            const double *x = &LTd[r * n];
            const double e = err[r];
            const std::array<double, 2> wd = aux_box_dot(x, inb->w_lo, inb->w_hi, n);
            const std::array<double, 2> gd = aux_box_dot(x, inb->g_lo, inb->g_hi, n);
            w_up[r] = (wd[0] + 0.0) + e * inb->beta_w;
            w_down[r] = (wd[1] + 0.0) - e * inb->beta_w;
            g_up[r] = (gd[0] + 0.0) + e * inb->beta_g;
            // the orders' |l' V_i,c| in their order, the remainder, the error
            double t1 = 0;
            for (int64_t i = 0; i < nv; ++i) {
                double rs = 0;
                for (int64_t c = k - 1; c >= 0; --c) rs = std::abs(lv[(r * nv + i) * k + c]) + rs;
                t1 = t1 + rs;
            }
            const double t2 = lrho[r];
            outer[r] = (t1 + t2) + e * inb->beta_u;
            double mig = 0;
            for (int64_t c = k - 1; c >= 0; --c) mig = std::abs(lvs[r * k + c]) + mig;
            inner[r] = (mig - t2) - e * inb->beta_u;
        }
    }
    // the sums over the steps before: cw, cwd (l'w), cov (outer), cin (inner), at k = 0 .. N
    const std::size_t rows_all = std::size_t(N + 1) * S;
    std::vector<double> cw(rows_all, 0.0), cwd(rows_all, 0.0), cov(rows_all, 0.0),
        cin(rows_all, 0.0);
    for (int64_t k = 0; k < N; ++k)
        for (int q = 0; q < S; ++q) {
            const std::size_t r = std::size_t(k) * S + q, next = r + S;
            cw[next] = cw[r] + w_up[r];
            cwd[next] = cwd[r] + w_down[r];
            cov[next] = cov[r] + outer[r];
            cin[next] = cin[r] + inner[r];
        }
    // falsification: the first time point k dt <= T, in order, whose lower distance is above 0
    const ExactTime step(dt);
    for (int64_t k = 1; k <= N; ++k)
        for (int q = 0; q < S; ++q) {
            const std::size_t r = std::size_t(k) * S + q;
            if (0 < ((sb.down[r] + cwd[r]) + cin[r]) - run.d[q] &&
                step.times(static_cast<std::uint32_t>(k)) <= T) {
                out.kind = SoundPass::Kind::falsified;
                out.k = k;
                out.q = q;
                return out;
            }
        }
    // the steps: the larger end plus the curvature terms and the outer input below 0
    out.flag.assign(S, std::vector<char>(N, 0));
    for (int64_t k = 0; k < N; ++k)
        for (int q = 0; q < S; ++q) {
            const std::size_t r = std::size_t(k) * S + q, next = r + S;
            const double t2 = aux_sum_up(mw, [&](int64_t j) { return std::abs(lg1[r * mw + j]); });
            const double cb = ((lw[r] + t2) + lr[r]) + err[r] * beta_f;
            const double a0 = (sb.up[r] + cw[r]) - run.d[q];
            const double a1 = (sb.up[next] + cw[next]) - run.d[q];
            out.flag[q][k] = (((std::max(a0, a1) + cb) + g_up[r]) + cov[next]) < 0;
        }
    return out;
}

// The passes --------------------------------------------------------------------------------------

/// B u_c and B G_U, each entry summed from the last index down (CORALean's brackets at nearest),
/// and whether either is not zero; no input without B.
SoundInput aux_sound_input(const LinearSys &sys, const Zonotope &U, int64_t n) {
    SoundInput in;
    in.u0.assign(n, 0.0);
    if (!sys.B()) return in;
    const std::vector<double> B = sys.B()->data(), uc = U.center().data();
    const std::vector<double> GU = U.generators().data();
    const int64_t q = sys.B()->shape()[1], k = U.generators().shape()[1];
    in.k = k;
    in.M.assign(std::size_t(n) * k, 0.0);
    for (int64_t l = 0; l < n; ++l) {
        const double *Bl = &B[std::size_t(l) * q];
        in.u0[l] = aux_sum_up(q, [&](int64_t j) { return Bl[j] * uc[j]; });
        for (int64_t c = 0; c < k; ++c)
            in.M[std::size_t(l) * k + c] =
                aux_sum_up(q, [&](int64_t j) { return Bl[j] * GU[std::size_t(j) * k + c]; });
    }
    for (double v : in.u0) in.has_u = in.has_u || v != 0;
    for (double v : in.M) in.has_m = in.has_m || v != 0;
    return in;
}

// The sound run -----------------------------------------------------------------------------------

/// LinearSys.verifySupportFuncFloat at nearest.
VerifyResult aux_verify_sound(const LinearSys &sys, const Zonotope &R0, const Zonotope &U,
                              double t_final, const SpecRows &rows, Expmat &E,
                              std::optional<double> kappa,
                              std::chrono::steady_clock::time_point start) {
    VerifyResult res;
    auto finish = [&]() {
        res.computation_time = priv_seconds_since(start);
        return res;
    };
    const Tensor &A = sys.A();
    const int64_t n = A.shape()[0];
    // the run's constants: the scales, the growth bound, |c| + sum_j |g_j|, the rows C_s C
    SoundRun run;
    run.S = rows.nr;
    run.s = E.scales;
    run.kappa = kappa.value_or(1.0);
    run.mu = kappa ? 1.0 / t_final : priv_verify_log_norm_up(A.data(), run.s, n);
    run.d = rows.ds;
    // |c| + sum_j |g_j| and its largest scaled entry
    const std::vector<double> c = R0.center().data(), G = R0.generators().data();
    const int64_t m0 = R0.generators().shape()[1];
    run.v.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        run.v[i] =
            std::abs(c[i]) + aux_sum_up(m0, [&](int64_t j) { return std::abs(G[i * m0 + j]); });
        run.beta_x = std::max(run.beta_x, run.v[i] / run.s[i]);
    }
    const Tensor Cs = Tensor::like(A, rows.Cs, {run.S, int64_t(rows.Cs.size()) / run.S});
    run.L = sys.C() ? Cs.matmul(*sys.C()) : Cs;
    run.E0.assign(run.S, 0.0); // the bracket of C_s C is a point at nearest
    run.in = aux_sound_input(sys, U, n);
    // the initial set
    {
        const std::vector<double> lc = run.L.matmul(R0.center()).data();
        const std::vector<double> lg = run.L.matmul(R0.generators()).data();
        const SupportBounds b0 = aux_support_bounds(lc, lg, m0, run.E0, run.beta_x);
        for (int q = 0; q < run.S; ++q)
            if (0 < b0.down[q] - run.d[q]) {
                res.falsification = Falsification{aux_extreme_x0(R0, lg, q), 0.0};
                return finish();
            }
    }
    double dt = aux_time_step(t_final / kStepsDiv, t_final);
    const ExactTime T(t_final);
    ExactTime tf = T;
    std::vector<ExactIntervals> unsat(run.S, ExactIntervals{{ExactTime(), T}});
    while (true) {
        // a pass of dt to tf: falsified, undecided, or its verified steps removed
        ++res.iterations;
        const int64_t N = aux_steps_to(tf, dt);
        res.time_step = dt;
        res.num_steps = static_cast<int>(N);
        const SoundPass pass = aux_sound_pass(A, E, run, R0, dt, N, T, E.sparse_a);
        if (pass.kind == SoundPass::Kind::not_converged) {
            dt *= kShrinkNonconverged;
            continue;
        }
        if (pass.kind == SoundPass::Kind::no_bound) return finish();
        if (pass.kind == SoundPass::Kind::falsified) {
            const ExactTime t = ExactTime(dt).times(static_cast<std::uint32_t>(pass.k));
            res.falsification = Falsification{
                aux_extreme_x0(R0, pass.ltg, std::size_t(pass.k) * run.S + pass.q), t.nearest()};
            return finish();
        }
        // the verified steps leave the unverified times
        bool all_verified = true;
        ExactTime last;
        for (int q = 0; q < run.S; ++q) {
            ExactTime t0;
            const ExactTime step(dt);
            for (int64_t k = 0; k < N; ++k) {
                const ExactTime t1 = t0 + step;
                if (pass.flag[q][k]) aux_setdiff_exact(unsat[q], t0, t1);
                t0 = t1;
            }
            // what is left, and its latest end: the next pass's horizon
            for (const auto &iv : unsat[q]) {
                all_verified = false;
                if (last < iv[1]) last = iv[1];
            }
        }
        if (all_verified) break;
        tf = last;
        dt *= kShrinkFixed;
        if (dt < numerics::kMinTimeStep) return finish();
    }
    res.verified = true;
    return finish();
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult priv_verify_ra_support_function(const LinearSys &sys, const Zonotope &R0,
                                             const Zonotope &U, double t_final,
                                             const std::vector<Specification> &specs) {
    const auto start = std::chrono::steady_clock::now();
    VerifyResult res;
    auto finish = [&]() {
        res.computation_time = priv_seconds_since(start);
        return res;
    };

    // The system and input: x' = A x + u, u in B U; u_trans = B center(U), GU = B generators(U).
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
    E.powers.emplace(A.data(), A.shape()[0]);
    E.scales = priv_verify_balance_scales(A.data(), A.shape()[0]);
    aux_init_sparse(E, A);
    // a system with a growth bound: the sound algorithm (others, #45)
    {
        const int64_t n = A.shape()[0];
        const std::vector<double> a = A.data();
        const std::optional<double> kappa =
            priv_verify_energy_growth(a, E.scales, n, 1.0 / t_final);
        if (kappa || std::max(priv_verify_log_norm_up(a, E.scales, n), 0.0) * t_final <= 1.0)
            return aux_verify_sound(sys, R0, U, t_final, aux_spec_rows(specs, false), E, kappa,
                                    start);
    }
    [[maybe_unused]] const FlushDenormals flush; // a no-op off x86
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
        // The interval matrices first, so a step they reject never pays for e^{A dt}.
        IntMat F, G;
        if (!aux_intmat(A, isu, E, time_step, F, G)) {
            time_step *= kShrinkNonconverged;
            continue;
        }
        const Tensor Delta = aux_expm(A, E, time_step);

        // The constant input solution; a slow sum shrinks the step.
        Tensor Pu = u.zeros_like();
        if (isu) {
            const std::optional<Tensor> P = aux_particular(A, u, E, Delta, time_step);
            if (!P) {
                time_step *= kShrinkNonconverged;
                continue;
            }
            Pu = *P;
        }

        // The affine solution at the time points: a positive distance is a hit.
        const Pass pass = aux_pass(R0, l_t, ds, isu, Delta, Pu, u, F, G, N, E.sparse_a);
        for (int k = 1; k <= N; ++k)
            for (int s = 0; s < S; ++s)
                if (pass.affine_tp[std::size_t(k) * S + s] > 0) {
                    res.falsification = Falsification{
                        aux_extreme_x0(R0, pass.ltg, std::size_t(k) * S + s), time_step * k};
                    return finish();
                }

        // Outer distance per step: the larger end point plus curvature and the particular solution.
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
