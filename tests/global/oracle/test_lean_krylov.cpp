// test_lean_krylov - the float Krylov subspace of CORALean against the oracle, bit for bit
//
// The Arnoldi iteration, the error bound with the order test and the order search of CORA.cpp's
// reach and verification with alg::Krylov (priv_krylov_float.cpp, priv_krylov_subspace.cpp) are
// CORALean's Krylov.arnoldiFloat, errorBoundFloat, orderOkFloat and subspaceFloat at dtype nearest:
// the same bits on random matrices, a non-normal one (entries 30 above the diagonal), a breakdown
// (the start vector in a 2-dimensional invariant subspace), a chain of 30 states and an order
// beyond 170, where CORA's k! overflows. The windows of the small system y' = T y are those of
// the standard reach of CORA.cpp on T, which agree with CORALean's up to the exponential (the
// lean backend has no expm): the time points to 1e-7, the time intervals inside CORALean's, whose
// curvature bounds the Taylor remainder by a norm (CORA's is elementwise, tighter by 1e-3).
//
// CORA: global/functions/helper/dynamics/contDynamics/linearSys/arnoldi.m
// CORA: contDynamics/@linearSys/private/priv_subspace_Krylov_individual_Jawecki.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cora/global/oracle/krylov.h"
#include "oracle_testing.h"

#include <tuple>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kBreakdown = 1e-12;
constexpr double kTol = 1e-9;

/// A problem: the matrix, the start vector and a name for the messages.
struct Problem {
    std::string name;
    Tensor A, c;
};

/// The basis vectors as the (n, k + 1) matrix and the Hessenberg data as the (k + 1, k) matrix.
Tensor aux_basis(const ArnoldiFloat &s, int64_t n, int64_t k) {
    std::vector<double> V(size_t(n * (k + 1)));
    for (int64_t j = 0; j <= k; ++j)
        for (int64_t r = 0; r < n; ++r) V[size_t(r * (k + 1) + j)] = s.V[size_t(j)][size_t(r)];
    return Tensor::from_data(V, {n, k + 1});
}
Tensor aux_hessenberg(const ArnoldiFloat &s, int64_t k) {
    std::vector<double> H(size_t((k + 1) * k), 0.0);
    for (int64_t j = 0; j < k; ++j)
        for (int64_t i = 0; i <= j + 1; ++i) H[size_t(i * k + j)] = s.H[size_t(j)][size_t(i)];
    return Tensor::from_data(H, {k + 1, k});
}

/// CORA.cpp's set lies in the oracle's box and is narrower by at most `gap` (relative).
void aux_inside(const Zonotope &mine, const lean::Zonotope &theirs, double gap,
                const std::string &what) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    double worst = 0;
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double slack = kTol * (1 + std::abs(b[i])), far = gap * (1 + std::abs(b[i]));
        worst = std::max(
            {worst, (a[i] - b[i]) / (1 + std::abs(b[i])), (d[i] - c[i]) / (1 + std::abs(d[i]))});
        ok = a[i] >= b[i] - slack && a[i] <= b[i] + far && c[i] <= d[i] + slack &&
             c[i] >= d[i] - far;
    }
    if (!ok) std::cerr << "  " << what << ": the widest difference " << worst << "\n";
    test::check(ok, what + ": inside the oracle's box and close to it");
}

/// The problems: random, non-normal, a breakdown, an order beyond 170.
std::vector<Problem> aux_problems(test::oracle::Random &rnd) {
    std::vector<Problem> out;
    out.push_back({"random", rnd.matrix(6, 6), rnd.vector(6)});
    // upper entries 30: far from normal, with a negative diagonal
    std::vector<double> nn(36, 0.0);
    for (int64_t i = 0; i < 6; ++i)
        for (int64_t j = i; j < 6; ++j) nn[size_t(i * 6 + j)] = i == j ? -1.0 - i : 30.0;
    out.push_back({"non-normal", Tensor::from_data(nn, {6, 6}), rnd.vector(6)});
    // A is block diagonal and c lives in the first block: Arnoldi breaks down after two passes
    std::vector<double> bd = rnd.matrix(6, 6).data();
    for (int64_t i = 0; i < 6; ++i)
        for (int64_t j = 0; j < 6; ++j)
            if ((i < 2) != (j < 2)) bd[size_t(i * 6 + j)] = 0.0;
    std::vector<double> c(6, 0.0);
    c[0] = 1.0, c[1] = rnd.scalar();
    out.push_back({"breakdown", Tensor::from_data(bd, {6, 6}), Tensor::from_data(c, {6, 1})});
    // order 180 on 6 states: passes beyond n run on zero vectors, the decay product has no k!
    out.push_back({"order 180", rnd.matrix(6, 6) * 0.2, rnd.vector(6)});
    // a chain of 30 states, for a longer Hessenberg matrix
    std::vector<double> chain(30 * 30, 0.0);
    for (int64_t i = 0; i < 30; ++i) {
        chain[size_t(i * 30 + i)] = -2.0;
        if (i > 0) chain[size_t(i * 30 + i - 1)] = 1.0 + 0.01 * rnd.scalar();
        if (i + 1 < 30) chain[size_t(i * 30 + i + 1)] = 1.0 + 0.01 * rnd.scalar();
    }
    out.push_back({"chain", Tensor::from_data(chain, {30, 30}), rnd.vector(30)});
    return out;
}

/// The stable systems of the weight: random shifted left, a non-normal one, a chain with a 2 x 2
/// block.
std::vector<Tensor> aux_stable(test::oracle::Random &rnd) {
    std::vector<Tensor> out;
    std::vector<double> a = rnd.matrix(7, 7).data();
    for (int64_t i = 0; i < 7; ++i) a[size_t(i * 7 + i)] -= 4.0;
    out.push_back(Tensor::from_data(a, {7, 7}));
    std::vector<double> nn(36, 0.0);
    for (int64_t i = 0; i < 6; ++i)
        for (int64_t j = i; j < 6; ++j) nn[size_t(i * 6 + j)] = i == j ? -1.0 - i : 30.0;
    out.push_back(Tensor::from_data(nn, {6, 6}));
    // complex eigenvalues: rotation blocks with a negative real part, coupled upward
    std::vector<double> rot(64, 0.0);
    for (int64_t b = 0; b < 4; ++b) {
        const int64_t i = 2 * b;
        rot[size_t(i * 8 + i)] = rot[size_t((i + 1) * 8 + i + 1)] = -0.5;
        rot[size_t(i * 8 + i + 1)] = 2.0 + b, rot[size_t((i + 1) * 8 + i)] = -2.0 - b;
        if (b < 3) rot[size_t(i * 8 + i + 2)] = 5.0;
    }
    out.push_back(Tensor::from_data(rot, {8, 8}));
    // row-dominant with a sparse pattern and off-diagonals of both signs
    std::vector<double> dom(81, 0.0);
    for (int64_t i = 0; i < 9; ++i) {
        dom[size_t(i * 9 + i)] = -3.0 - 0.1 * i;
        dom[size_t(i * 9 + (i + 1) % 9)] = 1.0 + 0.05 * i;
        dom[size_t(i * 9 + (i + 4) % 9)] = -0.7;
    }
    out.push_back(Tensor::from_data(dom, {9, 9}));
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;

    // Arnoldi: nu, the basis and the Hessenberg matrix after a number of passes
    test::oracle::run("krylov.arnoldi", [](test::oracle::Random &rnd) {
        for (const Problem &pr : aux_problems(rnd)) {
            const int64_t n = pr.A.shape()[0];
            const int passes = n > 10 ? 25 : 5;
            const lean::KrylovArnoldi r = lean::Krylov::arnoldi(pr.A, pr.c, passes, kBreakdown);
            const HostMatVec A = priv_state_matvec(LinearSys(pr.A));
            ArnoldiFloat s = priv_arnoldi_start(pr.c.data());
            priv_arnoldi_extend(s, A, passes, kBreakdown);
            test::oracle::check("nu " + pr.name, s.nu, r.nu);
            test::oracle::check("V " + pr.name, aux_basis(s, n, passes), r.V);
            test::oracle::check("H " + pr.name, aux_hessenberg(s, passes), r.H);
        }
    });

    // the bound: residuals, mu_A, mu_T, bound and test, for two tolerances
    test::oracle::run("krylov.errorBound", [](test::oracle::Random &rnd) {
        for (const Problem &pr : aux_problems(rnd)) {
            const int64_t n = pr.A.shape()[0];
            const int order = pr.name == "order 180" ? 180 : (n > 10 ? 25 : 5);
            const HostMatVec A = priv_state_matvec(LinearSys(pr.A));
            ArnoldiFloat s = priv_arnoldi_start(pr.c.data());
            priv_arnoldi_extend(s, A, order, kBreakdown);
            // the long problems once: each request reruns its passes in the oracle
            const bool long_run = order > 20;
            for (const double tol : {1e-3, 1e-300}) {
                if (long_run && tol < 1e-100) continue;
                const double tf = 0.5;
                const lean::KrylovBound r =
                    lean::Krylov::error_bound(pr.A, pr.c, order, tf, tol, kBreakdown);
                const std::string what = pr.name + " tol " + std::to_string(tol);
                const double mu_A = A.quad_bound();
                test::oracle::check("muA " + what, mu_A, r.mu_A);
                test::oracle::check("muT " + what, priv_hessenberg_quad_bound(s, order), r.mu_T);
                const std::vector<double> mine(s.residual.begin(), s.residual.begin() + order);
                test::oracle::check("residuals " + what, Tensor::from_data(mine, {1, order}),
                                    Tensor::from_data(r.residuals, {1, order}));
                test::oracle::check("bound " + what,
                                    priv_krylov_error_bound(s, mu_A, order, tf, 10), r.bound);
                test::oracle::check("ok " + what, priv_krylov_order_ok(s, mu_A, order, tf, tol, 10),
                                    r.ok);
            }
        }
    });

    // the order search: the order found, whether it was accepted, its bound and the data
    test::oracle::run("krylov.subspace", [](test::oracle::Random &rnd) {
        for (const Problem &pr : aux_problems(rnd)) {
            // the tolerance 1e-300 is never met: the order grows to n
            const int64_t n = pr.A.shape()[0];
            const HostMatVec A = priv_state_matvec(LinearSys(pr.A));
            const bool long_run = n > 10;
            for (const double tol : {1e-6, 1e-300}) {
                if (long_run && tol < 1e-100) continue;
                const double tf = 0.3;
                const int k0 = long_run ? 4 : 2, step = long_run ? 6 : 1;
                const lean::KrylovSubspace r =
                    lean::Krylov::subspace(pr.A, pr.c, k0, step, tf, tol, kBreakdown);
                // CORA.cpp's search on the same problem, then the same data in the same order
                const KrylovSearch s =
                    priv_krylov_search(A, pr.c.data(), k0, step, tf, tol, n + 1, 10, kBreakdown);
                const std::string what = pr.name + " tol " + std::to_string(tol);
                test::oracle::check("order " + what, s.order, int64_t(r.order));
                test::oracle::check("ok " + what, s.ok, r.ok);
                test::oracle::check("bound " + what, s.bound, r.bound);
                test::oracle::check("nu " + what, s.state.nu, r.nu);
                test::oracle::check("V " + what, aux_basis(s.state, n, s.order), r.V);
                test::oracle::check("H " + what, aux_hessenberg(s.state, s.order), r.H);
            }
        }
    });

    // the windows of y' = T y from e_1 against the standard reach of CORA.cpp on T
    if (test::oracle::serves("krylov.windows")) {
        test::oracle::op_name() = "krylov.windows";
        const lean::DTypeScope dtype("nearest");
        test::oracle::Random rnd(7);
        for (const Problem &pr : aux_problems(rnd)) {
            if (pr.A.shape()[0] > 10) continue;
            const int64_t n = pr.A.shape()[0];
            const HostMatVec A = priv_state_matvec(LinearSys(pr.A));
            const double tf = 0.4, dt = 0.1;
            const lean::KrylovWindows r =
                lean::Krylov::windows(pr.A, pr.c, 2, 1, tf, 1e-6, kBreakdown, dt, 16, 4);
            const KrylovSearch s =
                priv_krylov_search(A, pr.c.data(), 2, 1, tf, 1e-6, n + 1, 10, kBreakdown);
            test::check(s.ok == r.ok, "windows: ok " + pr.name);
            if (!r.ok || !r.time_interval) continue;
            // the small system T and the point e_1
            const int64_t k = s.order;
            std::vector<double> T(size_t(k * k), 0.0), e1(size_t(k), 0.0);
            for (int64_t j = 0; j < k; ++j)
                for (int64_t i = 0; i <= std::min(j + 1, k - 1); ++i)
                    T[size_t(i * k + j)] = s.state.H[size_t(j)][size_t(i)];
            e1[0] = 1.0;
            const LinearSys small(Tensor::from_data(T, {k, k}));
            const Zonotope X0(Tensor::from_data(e1, {k, 1}), Tensor::zeros({k, 0}));
            const ReachSet R = small.reach(
                {.initial_set = X0, .t_final = tf},
                alg::Standard{.time_step = dt, .taylor_terms = 16, .zonotope_order = 4});
            test::check(R.time_interval.size() == r.time_interval->size(),
                        "windows: the number of steps " + pr.name);
            // the time points to 1e-7, the time intervals inside the oracle's
            for (size_t i = 0; i < R.time_interval.size() && i < r.time_interval->size(); ++i) {
                aux_inside(R.time_interval[i], (*r.time_interval)[i], 1e-3,
                           "windows " + pr.name + " interval " + std::to_string(i));
                aux_inside(R.time_point[i], (*r.time_point)[i], 1e-7,
                           "windows " + pr.name + " point " + std::to_string(i));
            }
        }
    }

    // the Lyapunov weight: CORA.cpp's solve, checked by both, and the error carried in it
    test::oracle::run("krylov.lyapunovCertificate", [](test::oracle::Random &rnd) {
        for (const Tensor &A : aux_stable(rnd)) {
            const int64_t n = A.shape()[0];
            const SparseMatrix S = SparseMatrix::from_dense(A);
            for (const LyapunovMethod method : {LyapunovMethod::schur, LyapunovMethod::sign}) {
                const LyapunovWeight w = priv_lyapunov_solve(S, method);
                // the Schur solve always has a weight, the sign iteration where it converges
                if (method == LyapunovMethod::sign && w.P.empty()) continue;
                test::check(!w.P.empty(), "the solve of the weight is definite");
                const Tensor P = Tensor::from_data(w.P, {n, n}), L = Tensor::from_data(w.L, {n, n});
                // the accepted weight, one that is not symmetric and one the equation fails for
                std::vector<double> lop = w.P;
                lop[1] += 1e-9;
                const Tensor Pskew = Tensor::from_data(lop, {n, n});
                const Tensor identity = Tensor::eye(n);
                for (const auto &[name, Pc, Lc, a] :
                     {std::tuple{"solved", P, L, w.a}, std::tuple{"skew", Pskew, L, w.a},
                      std::tuple{"identity", identity, identity * 0.5, 0.5}}) {
                    // the oracle's check and CORA.cpp's, bit for bit
                    const lean::KrylovCertificate r =
                        lean::Krylov::lyapunov_certificate(A, Pc, Lc, a);
                    const LyapunovCertificate c =
                        priv_lyapunov_certificate(S, Pc.data(), Lc.data(), a);
                    // the same bits of the verdict, the bounds and the rate
                    const std::string what = std::to_string(n) + " " + name +
                                             (method == LyapunovMethod::sign ? " sign" : "");
                    if (std::string(name) == "solved" && method == LyapunovMethod::schur)
                        test::check(c.ok, "the solved weight is accepted");
                    test::oracle::check("ok " + what, c.ok, r.ok);
                    test::oracle::check("lower " + what, c.lower, r.lower);
                    test::oracle::check("upper " + what, c.upper, r.upper);
                    test::oracle::check("rate " + what, c.rate, r.rate);
                }
            }
        }
    });
    test::oracle::run("krylov.restartedBound", [](test::oracle::Random &rnd) {
        for (const Tensor &A : aux_stable(rnd)) {
            const int64_t n = A.shape()[0];
            const SparseMatrix S = SparseMatrix::from_dense(A);
            const LyapunovWeight w = priv_lyapunov_solve(S, LyapunovMethod::schur);
            const LyapunovCertificate c = priv_lyapunov_certificate(S, w.P, w.L, w.a);
            const Tensor P = Tensor::from_data(w.P, {n, n}), L = Tensor::from_data(w.L, {n, n});
            // five windows of different lengths and local errors
            const std::vector<double> dt{0.1, 0.25, 0.5, 0.05, 2.0},
                eps{1e-3, 2e-4, 5e-5, 1e-3, 1e-6};
            const lean::KrylovRestarted r = lean::Krylov::restarted_bound(
                A, P, L, w.a, Tensor::from_data(dt, {5, 1}), Tensor::from_data(eps, {5, 1}));
            RestartedBound carried(c);
            for (size_t j = 0; j < dt.size(); ++j) carried.step(dt[j], eps[j]);
            const std::string what = std::to_string(n);
            test::oracle::check("ok " + what, c.ok, r.ok);
            test::oracle::check("windows ok " + what, c.ok, r.windows_ok);
            test::oracle::check("bound " + what, carried.bound(), r.bound);
        }
    });

    return test::finish("lean krylov");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
