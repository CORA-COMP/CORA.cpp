// priv_verify_ra_krylov - reach-avoid verification of a high-dimensional system by Krylov supports
//
// Every halfspace a'y <= d of a specification becomes a direction l (a safe set: a; an unsafe set:
// -a, so that the set is avoided where the support function stays below -d). The support function
// of y = C x over x' = A x + B u, x(0) in R0, u in U along l is, with z(t) = e^{A' t} C' l, the sum
// of z(t)'c + sum |z(t)'g| over the center c and generators g of R0, plus the integral over [0, t]
// of the same sum for B U. z stays in the Krylov subspace of C' l under A', where only the products
// A' x are needed: z(t) = |C' l| V e^{H t} e_1. The step starts at t_final / 1000 and is halved
// until the specifications hold on every step or it is below min_time_step: all halfspaces of a
// safe set, one halfspace of an unsafe set per step (a sufficient test). The bounds are those of
// the Stepper below.
//
// Syntax:   res = priv_verify_ra_krylov(sys, R0, U, t_final, specs, alg);
// Inputs:   sys - the system, A dense or sparse
//           R0 - the initial set
//           U - the input set
//           t_final - the final time
//           specs - safe sets and unsafe sets
//           alg - the options
// Outputs:  res - verified (false if undecided at min_time_step), the passes, last step size, steps
// See also: LinearSys::verify, priv_reach_krylov
//
// CORA.cpp only: the ARCH-COMP26 AFF scripts rand04-06 call reachSuppFuncsKrylov, which is not in
// CORA v2026.1.0 (it is in the competition's copy); this follows the same scheme on the
// Krylov machinery of reach, with a support function bound per step, not a port of that file
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cora/cont_set/interval/interval.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Directions and norms ----------------------------------------------------------------------------

/// One direction l = C' (normal) with the support value its set must stay below.
struct Direction {
    Tensor W; ///< (q, k): the dot products of the vectors of R0 and B U with V, times |C' l|
    Tensor H; ///< (k, k): the Hessenberg matrix
    double limit = 0;   ///< the support function must stay below this
    bool strict = true; ///< below, else at most
    bool zero = false;  ///< the direction is zero: the support function is 0
    size_t group = 0;   ///< the specification of the direction: one of its rows must hold per step
};

/// The norms of the columns of the matrix, and their sums over the R0 columns and the input ones.
struct VectorNorms {
    double r0 = 0, input = 0;
};
VectorNorms aux_norms(const Tensor &columns, int64_t m0) {
    const int64_t rows = columns.shape()[0], cols = columns.shape()[1];
    const std::vector<double> v = columns.data();
    VectorNorms out;
    for (int64_t j = 0; j < cols; ++j) {
        double sq = 0.0;
        for (int64_t r = 0; r < rows; ++r) sq += v[size_t(r * cols + j)] * v[size_t(r * cols + j)];
        (j <= m0 ? out.r0 : out.input) += std::sqrt(sq);
    }
    return out;
}

// Bounds step by step -----------------------------------------------------------------------------

/// The bound of a direction step by step: the hull of the time points and the curvature F(H) e_1
/// (as alg::Krylov) for R0 and B U, the integral by the step times the integrand, the Krylov error.
class Stepper {
  public:
    Stepper(const Direction &dir, int64_t m0, const VectorNorms &norms, double dt,
            const alg::ReachAvoidKrylov &alg)
        : dir_(dir), m0_(m0), dt_(dt), error_r0_(alg.krylov_error * norms.r0),
          error_input_(alg.krylov_error * norms.input) {
        if (dir.zero) return;
        const int64_t k = dir.H.shape()[0];
        e_hdt_ = (dir.H * dt).expm();
        const Interval F = LinearSys(dir.H).correction_matrix_state(dt, alg.taylor_terms);
        F1_ = Interval(F.inf().slice(-1, 0, 1), F.sup().slice(-1, 0, 1));
        E_ = dir.H.eye_like(k);
        previous_ = dir.W.matmul(E_.slice(-1, 0, 1)).data();
    }

    /// limit - bound on the next step (positive: it holds); a zero direction has the bound 0.
    double next_margin() {
        if (dir_.zero) return dir_.limit;
        // the curvature acts on the state at the start of the step
        const Interval curvature = F1_->linear_map(E_).linear_map(dir_.W);
        E_ = e_hdt_.matmul(E_);
        const std::vector<double> current = dir_.W.matmul(E_.slice(-1, 0, 1)).data();
        const std::vector<double> c_inf = curvature.inf().data(), c_sup = curvature.sup().data();
        // the bound of the homogeneous part and of the integrand over this step
        double homogeneous = error_r0_, integrand = error_input_;
        for (size_t i = 0; i < current.size(); ++i) {
            const double lo = std::min(previous_[i], current[i]) + c_inf[i];
            const double hi = std::max(previous_[i], current[i]) + c_sup[i];
            // index 0 is the center of R0, 1..m0 its generators, m0 + 1 the input center
            const bool signed_term = i == 0 || int64_t(i) == m0_ + 1;
            const double term = signed_term ? hi : std::max(std::abs(lo), std::abs(hi));
            (int64_t(i) <= m0_ ? homogeneous : integrand) += term;
        }
        const double bound = homogeneous + integral_ + dt_ * std::max(integrand, 0.0);
        integral_ += dt_ * integrand;
        previous_ = current;
        return dir_.limit - bound;
    }

  private:
    const Direction &dir_;
    int64_t m0_;
    double dt_, error_r0_, error_input_, integral_ = 0.0;
    Tensor e_hdt_, E_;
    std::optional<Interval> F1_; ///< the first column of F; unset for a zero direction
    std::vector<double> previous_;
};

/// Whether every group has a direction that holds on every step of the pass: the steps in order,
/// stopping at the first step on which a group has none.
bool aux_pass_holds(const std::vector<Direction> &dirs, size_t groups, int64_t m0,
                    const VectorNorms &norms, double dt, int64_t steps,
                    const alg::ReachAvoidKrylov &alg) {
    std::vector<Stepper> steppers;
    steppers.reserve(dirs.size());
    for (const Direction &dir : dirs) steppers.emplace_back(dir, m0, norms, dt, alg);
    std::vector<char> held(groups);
    for (int64_t step = 1; step <= steps; ++step) {
        std::fill(held.begin(), held.end(), 0);
        for (size_t d = 0; d < dirs.size(); ++d) {
            const double margin = steppers[d].next_margin();
            held[dirs[d].group] |= dirs[d].strict ? margin > 0 : margin >= 0;
        }
        if (!std::all_of(held.begin(), held.end(), [](char c) { return c != 0; })) return false;
    }
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult priv_verify_ra_krylov(const LinearSys &sys, const Zonotope &R0, const Zonotope &U,
                                   double t_final, const std::vector<Specification> &specs,
                                   const alg::ReachAvoidKrylov &alg) {
    const auto start = std::chrono::steady_clock::now();
    VerifyResult res;
    const int64_t n = sys.dim();
    const Tensor &like = R0.center();

    // the vectors whose dot products with z(t) the support function needs, and their norms
    const CanonicalInput input = priv_canonical_input(sys, U);
    const int64_t m0 = R0.generators().shape().back();
    const Tensor vectors =
        Tensor::concat({R0.center(), R0.generators(), input.u_trans, input.GU}, -1);
    const VectorNorms norms = aux_norms(vectors, m0);
    const Tensor vectors_t = vectors.matrix_transpose();
    const LinearSys dual = sys.A_sparse()
                               ? LinearSys(sys.A_sparse()->transpose(), like.zeros_like({n, 1}))
                               : LinearSys(sys.A().matrix_transpose());
    const HostMatVec At = priv_state_matvec(dual);
    // Directions ----------------------------------------------------------------------------------
    // one direction C' l and its Krylov subspace per halfspace of the specifications
    std::vector<Direction> dirs;
    int64_t order = 15;
    size_t groups = 0;
    for (const Specification &spec : specs) {
        const bool unsafe = spec.type() == SpecType::unsafe_set;
        // an unsafe set is avoided where a halfspace of it is, every halfspace of a safe set holds
        const size_t first_group = groups;
        for (const SpecRow &row : priv_spec_rows(spec)) {
            Direction dir;
            dir.group = unsafe ? first_group : groups++;
            const double sign = unsafe ? -1.0 : 1.0;
            // the output direction l, and C' l in the state space
            std::vector<double> l(row.a);
            for (double &x : l) x *= sign;
            const Tensor lt = Tensor::like(like, l, {int64_t(l.size()), 1});
            const Tensor d = sys.C() ? sys.C()->matrix_transpose().matmul(lt) : lt;
            dir.limit = sign * row.b;
            dir.strict = unsafe;
            // the Krylov subspace of C' l under A' (none for the zero direction)
            const std::vector<double> dv = d.data();
            double sum = 0.0;
            for (double x : dv) sum += x * x;
            dir.zero = sum == 0.0;
            if (!dir.zero) {
                const KrylovSubspace S = priv_krylov_subspace(At, dv, order, alg.krylov_error,
                                                              alg.krylov_step, t_final, like);
                dir.W = vectors_t.matmul(S.V) * S.norm;
                dir.H = S.H;
            }
            dirs.push_back(std::move(dir));
        }
        if (unsafe) ++groups;
    }

    // the adaptive loop: the step is halved until every direction holds
    double dt = alg.time_step > 0 ? alg.time_step : t_final / 1000.0;
    while (dt >= alg.min_time_step) {
        ++res.iterations;
        const auto steps = int64_t(std::ceil(t_final / dt));
        res.time_step = dt;
        res.num_steps = int(std::min<int64_t>(steps, std::numeric_limits<int>::max()));
        if (aux_pass_holds(dirs, groups, m0, norms, dt, steps, alg)) {
            res.verified = true;
            break;
        }
        dt /= 2;
    }
    res.computation_time = priv_seconds_since(start);
    return res;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
