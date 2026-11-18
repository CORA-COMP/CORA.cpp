// test_lean_nonlinear_sys_reach_adaptive - reach with alg lin-adaptive against the CORALean oracle
//
// With the default linear step (closed_form, the oracle's linearStep = closedForm) a run of van der
// Pol is the oracle's bit for bit at nearest: the step sizes, the tensor orders, the sets over the
// steps and at the end, and the time, which is exact in both (ExactTime here, a rational there).
// The last step, whose size is CORALean's ratUp of the exact time left, is the oracle's step from
// CORA.cpp's state before it (nonlinearSys.linReachAdaptive). Short horizon: the run is the first
// steps of its 26 and the last one (the oracle's closed form takes 40-60 s a step; the steps do
// not depend on where the run stops); the tensor order stays 2 there, order 3 is the pass test's.
// With linear_step = cora (CORA's oneStep with expm, on the eigen backend) the steps agree to 1e-6
// (relative) and the sets lie in the oracle's, tighter by at most 2e-2.
//
// ORACLE: contDynamics/@nonlinearSys/reach_adaptive.m
// ORACLE: contDynamics/@nonlinearSys/linReach_adaptive.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       10-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/oracle/nonlinear_sys.h"
#include "global/exact_time.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kAgreement = 1e-6, kSets = 2e-2, kFinal = 0.015;
constexpr int kClosedSteps = 2, kCoraSteps = 10;

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x, const std::vector<Expr> &u) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0] + u[0]};
        },
        2, 1);
}

Zonotope start() { return {Tensor({1.4, 2.3}), Tensor({{0.05, 0.0}, {0.0, 0.05}})}; }

Zonotope input() { return {Tensor({0.0}), Tensor({{0.1}})}; }

/// The box of `mine` lies in the oracle's up to 1e-9 (relative) and is tighter by at most kSets.
void aux_inside(const std::string &what, const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i) {
        const double slack = 1e-9 * (1 + std::abs(b[i])), far = kSets * (1 + std::abs(b[i]));
        ok = a[i] >= b[i] - slack && a[i] <= b[i] + far && c[i] <= d[i] + slack &&
             c[i] >= d[i] - far;
    }
    test::check(ok, what + ": inside the oracle's box and close to it");
}

/// CORA's oneStep: the oracle's first steps to 1e-6, its sets enclosing CORA.cpp's.
void aux_cora_step() {
    const NonlinearSys sys = van_der_pol();
    alg::LinearizationAdaptive alg;
    alg.linear_step = alg::LinearStep::cora;
    const ReachSet mine = sys.reach(ReachProblem{start(), input(), kFinal}, alg);
    const lean::AdaptiveRun theirs = lean::NonlinearSys(sys).reach_adaptive(
        lean::Zonotope(start()), lean::Zonotope(input()), kFinal, alg, kCoraSteps);
    const auto K = static_cast<std::size_t>(kCoraSteps);
    test::check(theirs.steps.size() == K && mine.time_interval.size() >= K, "cora: the same steps");
    if (theirs.steps.size() != K || mine.time_interval.size() < K) return;
    bool same = true;
    for (std::size_t k = 0; k < K; ++k) {
        const double size = mine.time[k + 1] - mine.time[k];
        same = same && std::abs(size - theirs.steps[k]) <= kAgreement * theirs.steps[k];
    }
    test::check(same, "cora: the step sizes of the oracle");
    for (std::size_t k = 0; k < K; ++k)
        aux_inside("cora", mine.time_interval[k], theirs.time_interval[k]);
    aux_inside("cora", mine.time_point[K], theirs.time_point);
}

/// The default (closed_form): the oracle's first steps bit for bit.
void aux_closed_form() {
    const NonlinearSys sys = van_der_pol();
    const alg::LinearizationAdaptive alg;
    const ReachSet mine = sys.reach(ReachProblem{start(), input(), kFinal}, alg);
    const lean::AdaptiveRun theirs = lean::NonlinearSys(sys).reach_adaptive(
        lean::Zonotope(start()), lean::Zonotope(input()), kFinal, alg, kClosedSteps);
    const auto K = static_cast<std::size_t>(kClosedSteps);
    test::check(theirs.steps.size() == K && mine.time_interval.size() >= K, "the same steps");
    if (theirs.steps.size() != K || mine.time_interval.size() < K) return;
    // the times are the exact sums of the steps, rounded: the oracle's steps give the same ones
    ExactTime t;
    for (std::size_t k = 0; k < K; ++k) {
        const std::string at = " " + std::to_string(k);
        t = t + theirs.steps[k];
        test::oracle::check("time" + at, mine.time[k + 1], t.nearest());
        test::oracle::check("time interval" + at, mine.time_interval[k], theirs.time_interval[k]);
        test::check(theirs.tensor_orders[k] == 2, "the tensor order 2" + at);
    }
    test::oracle::check("time point", mine.time_point[K], theirs.time_point);
    test::oracle::check("end time", mine.time[K], theirs.t);
}

/// The state of CORA.cpp before a step, for the oracle.
lean::AdaptiveStepState aux_lean_state(const AdaptiveState &st) {
    // layout: #163 step 2
    const Tensor &like = detail::cora_center(st.R);
    lean::AdaptiveStepState out;
    out.i = st.i;
    out.t = st.t.text();
    out.time_step = st.time_step;
    out.finitehorizon = st.finitehorizon;
    out.varphi = st.varphi;
    out.err_adm_horizon = priv_column(like, st.err_adm_horizon);
    out.err_adm_deltatopt = priv_column(like, st.err_adm_deltatopt);
    out.tensor_order = st.tensor_order;
    out.minorder = st.minorder;
    out.orders.assign(st.orders.begin(), st.orders.end());
    out.zero_width.assign(st.zero_width.begin(), st.zero_width.end());
    out.kappa_deltat = st.kappa_deltat;
    out.kappa_abstrerr = st.kappa_abstrerr;
    out.steps = st.steps;
    return out;
}

/// The last step, which ends at t_final (CORALean's ratUp of the exact time left): the oracle's
/// step from CORA.cpp's state before it, bit for bit.
void aux_last_step() {
    const NonlinearSys sys = van_der_pol();
    const alg::LinearizationAdaptive alg;
    const std::optional<Zonotope> U = input();
    AdaptiveState st(start(), sys.dim(), 0.0);
    while (priv_adaptive_continues(st.steps, st.t, kFinal)) {
        const bool last = st.i > 1 && ExactTime(kFinal) - st.t < ExactTime(st.time_step);
        AdaptiveState next = priv_lin_reach_adaptive(sys, U, alg, kFinal, st);
        if (last) {
            const lean::AdaptiveStepResult theirs = lean::NonlinearSys(sys).lin_reach_adaptive(
                lean::Zonotope(input()), lean::Zonotope(st.R), aux_lean_state(st), kFinal, alg);
            test::oracle::check("last step", next.time_step, theirs.time_step);
            test::oracle::check("last time point", next.R, theirs.time_point);
            test::oracle::check("last time interval", next.Rti, theirs.time_interval);
            test::check(!priv_adaptive_continues(next.steps, next.t, kFinal), "the run ends there");
            return;
        }
        // the sets reduced for the next step, as reach does
        next.R = next.R.reduce_adaptive(alg.zeta_z);
        next.Rti = next.Rti.reduce_adaptive(alg.zeta_z * 5);
        st = next;
    }
    test::check(false, "the run has a last step");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("nonlinearSys.reachAdaptive",
                      [](test::oracle::Random &) { aux_closed_form(); });
    test::oracle::run("nonlinearSys.linReachAdaptive",
                      [](test::oracle::Random &) { aux_last_step(); });
    if (test::oracle::serves("nonlinearSys.reachAdaptive")) {
        // CORA's oneStep evaluates expm, which the lean backend has not
        const BackendScope eigen("eigen");
        const lean::DTypeScope dtype("nearest");
        aux_cora_step();
    }
    return test::finish("lean nonlinearSys reach adaptive");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
