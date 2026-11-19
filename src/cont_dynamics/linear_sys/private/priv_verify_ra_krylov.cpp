// priv_verify_ra_krylov - reach-avoid verification of a high-dimensional system by Krylov supports
//
// Every halfspace a'y <= d of a specification becomes a direction l (a safe set: a; an unsafe set:
// -a, avoided where the support function stays below -d). With z(t) = e^{A' t} C' l the support
// function of y = C x is z'c + sum |z'g| over R0 plus the integral of the same over B U; z lives in
// restart windows of Krylov subspaces of A' (KrylovWindows: their plan is tuning, each window
// CORALean's sound one). The decision is CORALean's LinearSys.verifyKrylovPassFloat: on each step
// every group needs a direction whose sound bound (priv_verify_krylov.h) stays below its limit. The
// step starts at t_final / 1000 and is halved until the pass holds, no finer step can help, or it
// is below min_time_step (tuning, like the plan); a pass that does not hold is undecided.
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
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "cont_dynamics/linear_sys/private/priv_krylov_rounding.h"
#include "cont_dynamics/linear_sys/private/priv_verify.h"
#include "cont_dynamics/linear_sys/private/priv_verify_krylov.h"
#include "cora/cont_set/interval/interval.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Directions --------------------------------------------------------------------------------------

/// One direction l = C' (normal) with the support value its set must stay below.
struct Direction {
    std::vector<double> w;             ///< C' l in the state space
    std::vector<KrylovWindow> windows; ///< the restart windows of z(t) = e^{A' t} C' l
    double limit = 0;                  ///< the support function must stay below this
    bool strict = true;                ///< below, else at most
    bool zero = false;                 ///< the direction is zero: the support function is 0
    size_t group = 0; ///< the specification of the direction: one of its rows must hold per step
};

// The pass ----------------------------------------------------------------------------------------

/// How a pass ends: every group holds on every step, or some step fails (for good if no finer step
/// size can help: the points and the subspace error alone exceed the limit of every direction).
enum class Pass { holds, fails, hopeless };

/// LinearSys.verifyKrylovPassFloat with CORA.cpp's tuning: the steps in order, stopping at the
/// first step on which a group has no direction whose sound bound holds; a bound that is not
/// computed fails the pass. Whether a finer step can help is tuning (the floor of the stepper).
Pass aux_pass(const std::vector<Direction> &dirs, size_t groups, const VerifyKrylovVectors &vec,
              double dt, int64_t ratio, int64_t steps) {
    std::vector<VerifyKrylovStepper> steppers;
    steppers.reserve(dirs.size());
    for (const Direction &dir : dirs) steppers.emplace_back(vec, dir.windows, ratio, dt);
    std::vector<char> held(groups), hope(groups);
    for (int64_t step = 1; step <= steps; ++step) {
        std::fill(held.begin(), held.end(), 0);
        std::fill(hope.begin(), hope.end(), 0);
        // a direction holds the step, or leaves hope that a finer step will
        for (size_t d = 0; d < dirs.size(); ++d) {
            const Direction &dir = dirs[d];
            if (dir.zero) {
                if (dir.strict ? 0.0 < dir.limit : 0.0 <= dir.limit) held[dir.group] = 1;
                hope[dir.group] = 1;
                continue;
            }
            const std::optional<double> b = steppers[d].next();
            if (!b) return Pass::fails;
            if (dir.strict ? *b < dir.limit : *b <= dir.limit) held[dir.group] = 1;
            const double floor = steppers[d].floor();
            if (dir.strict ? floor < dir.limit : floor <= dir.limit) hope[dir.group] = 1;
        }
        if (std::all_of(held.begin(), held.end(), [](char c) { return c != 0; })) continue;
        const bool hopeful = std::all_of(hope.begin(), hope.end(), [](char c) { return c != 0; });
        return hopeful ? Pass::fails : Pass::hopeless;
    }
    return Pass::holds;
}

} // namespace


// ===========================================  MAIN  =========================================== //

VerifyResult priv_verify_ra_krylov(const LinearSys &sys, const Zonotope &R0, const Zonotope &U,
                                   double t_final, const std::vector<Specification> &specs,
                                   const alg::ReachAvoidKrylov &alg, VerifyKrylovTrace *trace) {
    const auto start = std::chrono::steady_clock::now();
    VerifyResult res;
    const int64_t n = sys.dim();
    const Tensor &like = R0.center(); // the backend and the dtype

    // the vectors whose dot products with z(t) the support function needs: [c G] and [u GU]
    const CanonicalInput input = priv_canonical_input(sys, U);
    auto columns = [](const Tensor &M) {
        const int64_t r = M.shape()[0], c = M.shape()[1];
        const std::vector<double> d = M.data();
        std::vector<std::vector<double>> out(static_cast<size_t>(c),
                                             std::vector<double>(static_cast<size_t>(r)));
        for (int64_t j = 0; j < c; ++j)
            for (int64_t i = 0; i < r; ++i) out[size_t(j)][size_t(i)] = d[size_t(i * c + j)];
        return out;
    };
    const VerifyKrylovVectors vectors =
        priv_verify_krylov_vectors(R0.center().data(), columns(R0.generators().matrix_transpose()),
                                   input.u_trans.data(), columns(input.GU));
    const LinearSys dual = sys.A_sparse()
                               ? LinearSys(sys.A_sparse()->transpose(), like.zeros_like({n, 1}))
                               : LinearSys(sys.A().matrix_transpose());
    const HostMatVec At = priv_state_matvec(dual);
    // the first step size, the grid of the restart windows of every direction
    const double dt0 = alg.time_step > 0 ? alg.time_step : t_final / 1000.0;
    const auto steps0 = int64_t(std::ceil(t_final / dt0));
    // Directions ----------------------------------------------------------------------------------
    // one direction C' l and its restart windows per halfspace of the specifications
    std::vector<Direction> dirs;
    int64_t order = 15;
    size_t groups = 0;
    bool sound = true, weighted = false;
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
            dir.w = d.data();
            // LinearSys.verifyKrylovBoundsFloat's zero direction: every entry 0
            dir.zero = std::all_of(dir.w.begin(), dir.w.end(),
                                   [](double x) { return x <= 0.0 && 0.0 <= x; });
            if (!dir.zero) {
                // z(t) in restart windows of the first step size, a whole number of its steps each
                KrylovWindows trajectory(At, dir.w, dt0, steps0, alg.krylov_error, alg.krylov_step,
                                         order, like);
                while (!trajectory.done()) {
                    dir.windows.push_back(trajectory.next());
                    order = dir.windows.back().order;
                }
                sound = sound && trajectory.sound();
                weighted = weighted || trajectory.certificate().has_value();
            }
            dirs.push_back(std::move(dir));
        }
        if (unsafe) ++groups;
    }

    // the adaptive loop (tuning): the step is halved until every direction holds
    double dt = dt0;
    int64_t halvings = 0, ratio = 1, steps = steps0;
    double last_dt = dt;
    int64_t last_halvings = 0, last_ratio = 1;
    while (sound && dt >= alg.min_time_step) {
        ++res.iterations;
        last_dt = dt;
        last_halvings = halvings;
        last_ratio = ratio;
        steps = int64_t(std::ceil(t_final / dt));
        res.time_step = dt;
        res.num_steps = int(std::min<int64_t>(steps, std::numeric_limits<int>::max()));
        const Pass pass = aux_pass(dirs, groups, vectors, dt, ratio, steps);
        res.verified = pass == Pass::holds;
        if (pass != Pass::fails) break;
        dt = lean_half(dt);
        ratio *= 2;
        ++halvings;
    }
    res.computation_time = priv_seconds_since(start);
    // the last pass for the oracle: its directions, windows, bounds and the weight
    if (trace) {
        trace->groups = int64_t(groups);
        trace->halvings = last_halvings;
        trace->steps = steps;
        trace->dt0 = dt0;
        trace->verified = res.verified;
        if (weighted && At.lyapunov_slot->formed) {
            trace->P = At.lyapunov_slot->P;
            trace->L = At.lyapunov_slot->L;
            trace->a = At.lyapunov_slot->a;
        }
        // per direction: its plan, its windows and its bounds on the last pass
        for (const Direction &dir : dirs) {
            VerifyKrylovTraceDirection td{
                dir.w, {}, dir.limit, dir.strict, int64_t(dir.group), dir.windows, std::nullopt};
            for (const KrylovWindow &w : dir.windows) td.plan.emplace_back(w.order, w.steps);
            td.bounds = priv_verify_krylov_bounds(vectors, dir.zero, dir.windows, last_ratio,
                                                  last_dt, steps);
            trace->dirs.push_back(std::move(td));
        }
    }
    return res;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
